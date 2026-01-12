import pytest
from PyQt6.QtWidgets import QApplication, QPushButton, QFileDialog
from PyQt6.QtCore import Qt
import sys
import os

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Swarm2d.simulation_gui import SwarmSimGUI
from Swarm2d.gui_config import get_default_config
import json
from unittest.mock import patch

@pytest.fixture
def app(qtbot):
    """Create a fixture for the GUI application."""
    test_app = QApplication.instance() or QApplication([])
    window = SwarmSimGUI()
    qtbot.addWidget(window)
    window.show()
    return window

def find_button(app, text):
    """Helper function to find a button by its text."""
    buttons = app.findChildren(QPushButton)
    for button in buttons:
        if button.text() == text:
            return button
    return None

def test_start_and_stop_simulation(app, qtbot):
    """Tests the Start and Stop buttons."""
    # Test Start button
    start_button = find_button(app, "Start")
    assert start_button is not None, "Could not find 'Start' button."
    
    assert app.sim_thread is None
    qtbot.mouseClick(start_button, Qt.MouseButton.LeftButton)
    
    assert app.sim_thread is not None
    assert app.sim_thread.isRunning()
    assert "Starting" in app.status_bar.currentMessage()

    # Test Stop button
    stop_button = find_button(app, "Stop")
    assert stop_button is not None, "Could not find 'Stop' button."
    
    qtbot.mouseClick(stop_button, Qt.MouseButton.LeftButton)
    app.sim_thread.wait(1000)  # Wait for the thread to finish
    
    assert not app.sim_thread.isRunning()
    assert "Stopping" in app.status_bar.currentMessage()

def test_save_preset(app, qtbot, tmp_path):
    """Tests the Save Preset button by mocking the file dialog."""
    save_button = find_button(app, "Save Preset")
    assert save_button is not None, "Could not find 'Save Preset' button."

    # Create a temporary file path for the preset
    preset_path = os.path.join(tmp_path, "test_preset.json")

    with patch.object(QFileDialog, 'getSaveFileName', return_value=(preset_path, '')):
        qtbot.mouseClick(save_button, Qt.MouseButton.LeftButton)

    assert os.path.exists(preset_path)
    with open(preset_path, 'r') as f:
        saved_data = json.load(f)
    
    assert "base_config" in saved_data
    assert "team_overrides" in saved_data
    assert saved_data["base_config"] == app.base_config

def test_load_preset(app, qtbot, tmp_path):
    """Tests the Load Preset button by mocking the file dialog."""
    load_button = find_button(app, "Load Preset")
    assert load_button is not None, "Could not find 'Load Preset' button."

    # Create a dummy preset file
    preset_path = os.path.join(tmp_path, "load_test.json")
    dummy_config = {
        "base_config": {
            "General": {
                "num_teams": {"value": 5, "min": 1, "max": 6, "type": "int"},
                "num_agents_per_team": {"value": 15, "min": 1, "max": 50, "type": "int"}
            }
        },
        "team_overrides": {}
    }
    with open(preset_path, 'w') as f:
        json.dump(dummy_config, f)

    with patch.object(QFileDialog, 'getOpenFileName', return_value=(preset_path, '')):
        qtbot.mouseClick(load_button, Qt.MouseButton.LeftButton)
    
    assert app.base_config["General"]["num_teams"]["value"] == 5
    assert app.base_config["General"]["num_agents_per_team"]["value"] == 15
    assert app.team_selector.count() == 6 # 5 teams + global

def get_all_param_keys():
    """Recursively finds all parameter keys in the config dictionary."""
    config = get_default_config()
    keys = []
    
    def recurse_find_keys(d):
        for key, details in d.items():
            if isinstance(details, dict):
                if 'value' in details:
                    keys.append(key)
                else:
                    recurse_find_keys(details)
    
    recurse_find_keys(config)
    return keys

@pytest.mark.parametrize("param_name", get_all_param_keys())
def test_all_parameter_changes(app, qtbot, param_name):
    """
    Tests that changing any parameter widget in the GUI updates the internal config.
    This test is parametrized to run for every parameter found in the config.
    """
    widget, details = app.find_widget_and_details(param_name)
    assert widget is not None, f"Could not find widget for parameter '{param_name}'"
    assert details is not None, f"Could not find config details for parameter '{param_name}'"

    param_type = details.get('type')
    new_value = None

    if param_type == 'bool':
        original_value = widget.isChecked()
        new_value = not original_value
        widget.toggle()  # More reliable for tests than simulating a click

    elif param_type in ['int', 'float', 'reward']:
        slider = widget.get('slider') or widget.get('base_slider')
        
        # Change the value by a small amount, respecting bounds
        if slider.value() < slider.maximum():
            slider.setValue(slider.value() + 1)
        else:
            slider.setValue(slider.value() - 1)
        
        new_value = app.get_value_from_widget(widget, details)

    elif param_type == 'randomizable':
        base_slider = widget['base_slider']
        if base_slider.value() < base_slider.maximum():
            base_slider.setValue(base_slider.value() + 1)
        else:
            base_slider.setValue(base_slider.value() - 1)
            
        new_value = app.get_value_from_widget(widget, details)

    else:
        pytest.skip(f"Test not implemented for parameter type '{param_type}' for '{param_name}'")

    # The config is updated in place, so we re-fetch the details to check the value
    _, updated_details = app.find_widget_and_details(param_name)
    updated_value = updated_details['value']

    if param_type == 'randomizable':
        assert updated_value['base'] == pytest.approx(new_value['base'])
        assert updated_value['rand'] == pytest.approx(new_value['rand'])
    elif isinstance(new_value, float):
        assert updated_value == pytest.approx(new_value)
    else:
        assert updated_value == new_value

def test_num_teams_slider_updates_config_and_selector(app, qtbot):
    """
    Tests if moving the 'num_teams' slider updates both the internal
    configuration and the team selector dropdown.
    """
    # 1. Find the 'num_teams' widget using the helper function
    widget_dict, _ = app.find_widget_and_details("num_teams")
    assert widget_dict is not None, "Could not find the 'num_teams' widget."
    slider = widget_dict.get('slider')
    assert slider is not None, "'num_teams' widget does not have a slider component."

    # 2. Simulate moving the slider to a new value
    new_team_count = 4
    qtbot.mouseClick(slider, Qt.MouseButton.LeftButton) # Needed to give focus
    slider.setValue(new_team_count)

    # 3. Verify the internal base_config is updated
    # The on_parameter_changed signal should have fired and updated the config
    assert app.base_config['General']['num_teams']['value'] == new_team_count, \
        f"Internal config was not updated. Expected {new_team_count}, got {app.base_config['General']['num_teams']['value']}"

    # 4. Verify the team selector dropdown has the correct number of items
    # It should have one item for "Global Defaults" plus one for each team
    expected_item_count = new_team_count + 1
    assert app.team_selector.count() == expected_item_count, \
        f"Team selector has wrong number of items. Expected {expected_item_count}, got {app.team_selector.count()}"

def test_load_defaults_resets_parameters(app, qtbot):
    """
    Tests if clicking the 'Load Defaults' button resets a modified parameter.
    """
    # 1. Find a widget and change its value from the default
    num_agents_widget, _ = app.find_widget_and_details("num_agents_per_team")
    slider = num_agents_widget.get('slider')
    
    original_value = app.base_config['General']['num_agents_per_team']['value']
    new_value = original_value + 5
    slider.setValue(new_value)
    
    assert app.base_config['General']['num_agents_per_team']['value'] == new_value

    # 2. Find and click the 'Load Defaults' button
    defaults_button = find_button(app, "Load Defaults")
    assert defaults_button is not None, "Could not find 'Load Defaults' button."
    qtbot.mouseClick(defaults_button, Qt.MouseButton.LeftButton)

    # 3. Verify the parameter has been reset to its default value
    assert app.base_config['General']['num_agents_per_team']['value'] == original_value, \
        "Parameter was not reset to default after loading defaults."
