import pygame
import numpy as np
import math
import torch.nn as nn

from constants import *
from env.helper import normalize_vector

class RenderManager (nn.Module):
    

    def render(self, mode="human", suppress_overlay=False):
        """Renders the environment using Pygame."""
        if not self.render_mode or mode != "human":
            return

        # --- Robust Pygame/Font Initialization ---
        # Ensure Pygame display is initialized
        if self.screen is None:
            try:
                pygame.init() # Initialize all Pygame modules
                pygame.display.init()
                pygame.font.init() # Explicitly initialize font module
                self.screen = pygame.display.set_mode((DISP_WIDTH, DISP_HEIGHT))
                pygame.display.set_caption("Swarm2DEnv")
                self.clock = pygame.time.Clock()
                # Initialize font attribute ONCE here after successful init
                try:
                    self.font = pygame.font.SysFont("arial", 14)
                except pygame.error:
                    print("Warning: SysFont 'arial' not found, using default font.")
                    self.font = pygame.font.Font(None, 18) # Fallback font

            except pygame.error as e:
                print(f"Error initializing Pygame display/font: {e}")
                self.render_mode = False
                try: pygame.quit() # Clean up if init failed
                except: pass
                return # Exit render if init fails

        # Handle Pygame events (like closing the window)
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.render_mode = False # Stop rendering loop
                    pygame.quit() # Cleanly quit Pygame
                    return # Exit render function
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.render_mode = False
                    pygame.quit()
                    return
        except pygame.error as e: # Catch errors if display closed unexpectedly
             print(f"Pygame event handling error: {e}")
             self.render_mode = False
             return

        # --- Drawing ---
        self.screen.fill((30, 30, 30)) # Dark background

        # Scaling factors (calculated dynamically if needed, or use fixed)
        scale_x = DISP_WIDTH / self.width
        scale_y = DISP_HEIGHT / self.height

        # --- Get Font (Ensure font is initialized) ---
        render_font = self.font if hasattr(self, 'font') and self.font else None

        # Draw explored points (optional, can be slow)
        for pt in self.explored_points:
             x = int(pt[0] * scale_x)
             y = int(pt[1] * scale_y)
             pygame.draw.circle(self.screen, (40, 40, 40), (x, y), 1)

        # --- Get Team Names for Rendering ---
        if self.env.team_configs:
            team_names_render = {i: team['name'] for i, team in enumerate(self.env.team_configs)}
        else:
            team_names_render = {i: f"Team {i}" for i in range(self.env.num_teams)}

        # Draw Hives
        for team_idx, hive in self.hives.items():
            if hive and hive.get('pos') is not None:
                pos = hive["pos"]
                max_hive_health = self.metadata.get('hive_max_health', HIVE_MAX_HEALTH) # Use metadata default
                health_frac = max(0, min(1, hive["health"] / max_hive_health)) if max_hive_health > 0 else 1.0

                # Determine Owner and Original Team ID
                current_owner = hive.get('owner', team_idx) # Default to original if owner key missing
                original_team = team_idx # The dictionary key is the original team

                # --- Calculate Fill Color (Based on CURRENT OWNER) ---
                owner_color_rgba = TEAM_COLORS.get(current_owner, [128, 128, 128, 255]) # Grey fallback
                # Fade fill color based on health (e.g., towards a darker grey)
                dark_grey = 80
                lerp_r = int(owner_color_rgba[0] * health_frac + dark_grey * (1 - health_frac))
                lerp_g = int(owner_color_rgba[1] * health_frac + dark_grey * (1 - health_frac))
                lerp_b = int(owner_color_rgba[2] * health_frac + dark_grey * (1 - health_frac))
                hive_fill_color = tuple(np.clip([lerp_r, lerp_g, lerp_b], 0, 255))

                # --- Draw Fill ---
                x = int(pos[0] * scale_x)
                y = int(pos[1] * scale_y)
                # Use assumed radius from metadata for consistent drawing size
                hive_render_radius = int(self.metadata.get('hive_radius_assumed', 25.0) * scale_x)
                pygame.draw.circle(self.screen, hive_fill_color, (x, y), hive_render_radius)

                # --- Determine and Draw Border Color (Based on Ownership Status) ---
                if current_owner != original_team:
                    # If CAPTURED, use a distinct "captured" border color
                    border_color = (211, 211, 211) # Light Grey border for captured
                    border_thickness = 3 # Make captured border thicker
                else:
                    # If owned by ORIGINAL team, use original team color for border
                    original_team_color_rgb = TEAM_COLORS.get(original_team, [255, 255, 255])[:3] # Fallback white
                    border_color = original_team_color_rgb
                    border_thickness = 2 # Standard thickness

                # Draw the border
                pygame.draw.circle(self.screen, border_color, (x, y), hive_render_radius + 1, border_thickness)

                if render_font:
                    try:
                        team_name = team_names_render.get(team_idx, f"T{team_idx}")
                        # Use black text for better contrast maybe? Or white? Let's try white.
                        text_color = (255, 255, 255)
                        text_surf = render_font.render(team_name, True, text_color)
                        text_rect = text_surf.get_rect()
                        # Position text centered above the hive circle
                        text_rect.center = (x, y - hive_render_radius - 8) # Adjust vertical offset (8 pixels above border)
                        self.screen.blit(text_surf, text_rect)
                    except Exception as e_text:
                        pass # Continue rendering other elements

        # Draw Resources (using stored radius_pb and self.max_radius_pb)
        for res in self.resources:
            if res and not res.get('delivered') and res.get('pos') is not None:
                pos = res['pos']
                res_radius_pb = res.get('radius_pb')
                # *** Use self.max_radius_pb for clipping ***
                render_radius = int(np.clip(res_radius_pb * scale_x, 2, self.max_resource_radius_pb * scale_x))
                # --- End Use self.max_radius_pb ---
                color = (255, 165, 0) if res["cooperative"] else (0, 255, 0)
                pygame.draw.circle(self.screen, color,
                                (int(pos[0] * scale_x), int(pos[1] * scale_y)),
                                render_radius)
        # Draw Obstacles
        for obs in self.obstacles:
            if obs and obs.get('pos') is not None:
                rect_size = int(2 * obs['radius'] * scale_x)
                top_left_x = int((obs['pos'][0] - obs['radius']) * scale_x)
                top_left_y = int((obs['pos'][1] - obs['radius']) * scale_y)
                pygame.draw.rect(self.screen, (128, 128, 128), (top_left_x, top_left_y, rect_size, rect_size))

        # --- Draw Agent Observation Radii (Revised for Layering) ---
        for i, agent in enumerate(self.agents):
            if agent and agent.get('alive') and agent.get('pos') is not None:
                # Check if we should render this agent's radius
                # If single_agent_obs_idx is set, only render that one.
                target_idx = getattr(self.env, 'single_agent_obs_idx', None)
                if target_idx is not None and i != target_idx:
                    continue

                center_x = int(agent['pos'][0] * scale_x)
                center_y = int(agent['pos'][1] * scale_y)
                obs_radius_pixels = int(agent['obs_radius'] * scale_x)

                if obs_radius_pixels > 0:
                    temp_surf = pygame.Surface((DISP_WIDTH, DISP_HEIGHT), pygame.SRCALPHA)
                    circle_color_rgba = (200, 200, 0, 40) # Yellowish, low alpha
                    pygame.draw.circle(temp_surf, circle_color_rgba,
                                       (center_x, center_y), obs_radius_pixels, 0) # width=0 for filled
                    self.screen.blit(temp_surf, (0, 0))

        # --- Draw Fading Agent Trails ---
        for agent_idx, trail in enumerate(self.agent_trails):
            agent = self.agents[agent_idx]
            if not agent or not agent.get('alive'):
                continue
            
            agent_team = agent['team']
            base_color_rgba = TEAM_COLORS.get(agent_team, [200, 200, 200, 255])
            base_color_rgb = base_color_rgba[:3]

            num_points = len(trail)
            for i, pos in enumerate(trail):
                # Calculate alpha: fades from 150 to 0
                alpha = int(150 * (i / num_points))
                
                # Create a color with fading alpha
                trail_color = (*base_color_rgb, alpha)

                trail_surf = pygame.Surface((DISP_WIDTH, DISP_HEIGHT), pygame.SRCALPHA)

                x = int(pos[0] * scale_x)
                y = int(pos[1] * scale_y)
                
                pygame.draw.circle(trail_surf, trail_color, (x, y), 2)

                self.screen.blit(trail_surf, (0, 0))

        # Draw Agents
        MAAC_ROLE_COLORS = { # Keep role colors for MAAC teams
            "scout": (255, 255, 0), "collector": (0, 255, 255),
            "defender": (255, 0, 255), "attacker": (255, 165, 0)
        }
        agent_render_radius = int(AGENT_RADIUS * scale_x)
        for agent_idx, agent in enumerate(self.agents):
            if not agent or not agent.get('alive', False) or agent.get('pos') is None: continue

            agent_team = agent['team']
            # Inside agent drawing loop:
            base_color_rgba = TEAM_COLORS.get(agent_team, [200, 200, 200, 255]) # Grey fallback
            final_color_rgb = list(base_color_rgba[:3]) # Use first 3 elements

            # Inside hive drawing loop (fill color):
            owner_color_rgba = TEAM_COLORS.get(current_owner, [128, 128, 128, 255]) # Grey fallback
            # ... lerp calculation uses owner_color_rgba[0], [1], [2] ...

            # Inside hive drawing loop (border color):
            original_team_color_rgb = TEAM_COLORS.get(original_team, [255, 255, 255, 255])[:3] # Use first 3 elements, White fallback
            border_color = original_team_color_rgb # If owner==original
            # Check for MAAC teams (T0 or T3) and apply role color if available
            is_maac_team = (agent_team % 3 == 0) # MAAC is team 0 and 3 (assuming L/G split)
            if is_maac_team:
                # Use agent's GLOBAL ID (index in self.agents) to lookup role
                assigned_role = self.current_maac_roles.get(agent_idx) # Use agent index as key
                if assigned_role and assigned_role in MAAC_ROLE_COLORS:
                    final_color_rgb = list(MAAC_ROLE_COLORS[assigned_role]) # Override color

            # Adjust color based on health (fade to darker shade?)
            health_frac = max(0, min(1, agent['health'] / agent['max_health'])) if agent['max_health'] > 0 else 1.0
            final_color_rgb = [int(c * (0.5 + 0.5 * health_frac)) for c in final_color_rgb] # Fade to half brightness at 0 health
            # Apply slowed effect visual (e.g., make slightly grey?)
            if agent.get('slowed_timer', 0) > 0:
                final_color_rgb = [int(c * 0.6 + 100 * 0.4) for c in final_color_rgb] # Blend towards grey
            x = int(agent['pos'][0] * scale_x)
            y = int(agent['pos'][1] * scale_y)
            pygame.draw.circle(self.screen, final_color_rgb, (x, y), agent_render_radius)
            pygame.draw.circle(self.screen, (0,0,0), (x, y), agent_render_radius, 1) # Black border

            # --- Draw energy bar below agent (MODIFIED) ---
            energy_frac = max(0, min(1, agent['energy'] / agent['max_energy'])) if agent['max_energy'] > 0 else 0.0
            bar_width = int(agent_render_radius * 7) # Make slightly longer
            bar_height = 2 # Make slightly thicker
            bar_x = x - bar_width // 2 # Center the bar
            bar_y = y + agent_render_radius + 4 # Increase gap below agent
            # Draw background bar (grey)
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            # Draw energy fill (BLUE color, distinct from green resources)
            energy_bar_color = (0, 100, 255) # Blue color for energy
            pygame.draw.rect(self.screen, energy_bar_color, (bar_x, bar_y, int(bar_width * energy_frac), bar_height))
            
            # --- Add Health Bar (Also Wider) ---
            health_frac = max(0, min(1, agent['health'] / agent['max_health'])) if agent['max_health'] > 0 else 0.0
            health_bar_y = bar_y + bar_height + 1 # Position below energy bar
            health_bar_color = (255, 50, 50) # Red color for health
            # Draw background
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, health_bar_y, bar_width, bar_height)) # Use same wider width
            # Draw health fill
            pygame.draw.rect(self.screen, health_bar_color, (bar_x, health_bar_y, int(bar_width * health_frac), bar_height))


            if 'vel' in agent and agent['vel'] is not None:
                vel_x, vel_y = agent['vel']
                raw_speed_sq = vel_x**2 + vel_y**2

                # Only draw if moving noticeably
                speed_threshold_sq = 0.1 # Adjust threshold as needed
                if raw_speed_sq > speed_threshold_sq:
                    raw_speed = math.sqrt(raw_speed_sq)
                    # Get max_speed for normalization (use observed if available)
                    # Ensure self.metadata is accessible here
                    default_for_render_norm = self.metadata.get('bee_speed') * 1.2 # More consistent default
                    max_speed_norm = self.metadata.get('max_agent_speed_observed', default_for_render_norm)
                    max_speed_norm = max(max_speed_norm, 1e-6) # Avoid division by zero
                    # Normalize the speed magnitude (cap at 1.0)
                    normalized_speed = min(raw_speed / max_speed_norm, 1.0)
                    # Define the maximum pixel length for the line (when speed is max)
                    max_line_pixel_length = 10 # Adjust for desired max length
                    # Calculate the actual line length based on normalized speed
                    current_line_length = normalized_speed * max_line_pixel_length
                    # Calculate the direction unit vector
                    dir_x = vel_x / raw_speed
                    dir_y = vel_y / raw_speed
                    # Calculate the endpoint
                    end_x = int(x + dir_x * current_line_length)
                    end_y = int(y + dir_y * current_line_length)
                    # Draw the line
                    pygame.draw.line(self.screen, (255, 255, 255), (x, y), (end_x, end_y), 2)
        # --- Render Textual Info ---
        try:
            font = pygame.font.SysFont("arial", 14)
        except pygame.error:
            font = pygame.font.Font(None, 18) # Fallback font

        text_lines = []
        # Team names for rendering (adjust if team structure changes)
        if self.env.team_configs:
            team_names_render = {i: team['name'] for i, team in enumerate(self.env.team_configs)}
        else:
            team_names_render = {i: f"Team {i}" for i in range(self.env.num_teams)}

        # Calculate and display average team energy & agent count
        team_avg_energies = {}
        team_agent_counts = {}
        for team in range(self.num_teams):
            team_agents = [agent for agent in self.agents if agent and agent["team"] == team and agent["alive"]]
            team_agent_counts[team] = len(team_agents)
            team_avg = np.mean([agent["energy"] for agent in team_agents]) if team_agents else 0.0
            team_avg_energies[team] = team_avg

        for team in range(self.num_teams):
            team_name = team_names_render.get(team, f"Team {team}")
            hive = self.hives.get(team)
            hive_health_str = f"{hive['health']:.0f}" if hive else "N/A"
            hive_owner_str = f"(Own: T{hive['owner']})" if hive and hive['owner'] != team else ""
            energy_str = f"{team_avg_energies[team]:.0f}"
            count_str = f"{team_agent_counts[team]}"
            text_lines.append(f"{team_name}: H {hive_health_str}{hive_owner_str}, E {energy_str}, N {count_str}")

        text_lines.append(f"Picked: {self.resources_picked_count}, Delivered: {self.resources_delivered_count}")
        text_lines.append(f"Kills: {self.agents_killed_count}")
        text_lines.append(f"Step: {self.step_counter}/{self.max_steps}")

        # Now render the lines using render_font
        if render_font and not suppress_overlay: # Check if font object is valid and overlay not suppressed
            y_offset = 10
            for line in text_lines:
                try:
                    text_surface = render_font.render(line, True, (255, 255, 255))
                    self.screen.blit(text_surface, (10, y_offset))
                    y_offset += 18 # Spacing
                except pygame.error as e:
                    print(f"Warning: Pygame font rendering error (info block): {e}")
                    break # Stop trying if error occurs
        else:
            # This warning should ideally not print anymore with the robust init
            print("Warning: Font object (render_font) not available for info block rendering.")


        # --- Update Display ---
        try:
            pygame.display.flip()
            if self.clock: self.clock.tick(FPS) # Cap frame rate
            return True # Return True to indicate successful render
        except pygame.error as e:
            print(f"Pygame display flip/tick error: {e}")
            self.render_mode = False # Stop rendering if display fails
            return False

    def get_frame_array(self, suppress_overlay=False):
        """Returns the current Pygame screen as a RGB numpy array for external visualization."""
        if self.screen is None:
            # Initialize an off-screen surface if no screen exists
            self.screen = pygame.Surface((DISP_WIDTH, DISP_HEIGHT))
        
        # Run the standard render logic to draw everything onto self.screen
        self.render(mode="human", suppress_overlay=suppress_overlay) 
        
        # Convert Pygame surface to 3D RGB array
        view = pygame.surfarray.array3d(self.screen)
        # Pygame uses (width, height, rgb), Matplotlib/PIL use (height, width, rgb)
        return np.transpose(view, (1, 0, 2))