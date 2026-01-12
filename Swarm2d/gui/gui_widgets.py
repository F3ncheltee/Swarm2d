from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QCheckBox, QLineEdit, QMessageBox
from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6.QtCore import Qt

class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None, help_text=""):
        super(CollapsibleBox, self).__init__(parent)

        header_layout = QHBoxLayout()
        
        self.toggle_button = QPushButton(title)
        self.toggle_button.setStyleSheet("text-align: left; font-weight: bold;")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        header_layout.addWidget(self.toggle_button)

        if help_text:
            self.help_button = QPushButton("?")
            self.help_button.setFixedSize(25, 25)
            self.help_button.clicked.connect(lambda: QMessageBox.information(self, "Help", help_text))
            header_layout.addWidget(self.help_button)

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setSpacing(2)
        self.content_area.hide()

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(header_layout)
        layout.addWidget(self.content_area)

        self.toggle_button.toggled.connect(self.toggle)

    def toggle(self, checked):
        if checked:
            self.content_area.show()
        else:
            self.content_area.hide()

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

def create_parameter_widget(name, params, base_value=None, is_team_specific=False):
    """Factory function to create a widget based on parameter type."""
    widget_type = params.get("type")
    if widget_type == "int" or widget_type == "float":
        return create_slider(name, params)
    elif widget_type == "bool":
        return create_checkbox(name, params)
    elif widget_type == "reward":
        # Pass the base_value and layout specifier to the reward slider creator
        return create_reward_slider(name, params, base_value, is_team_specific=is_team_specific)
    elif widget_type == "randomizable":
        return create_randomizable_slider(name, params)
    
    unsupported_label = QLabel(f"Unsupported type: {widget_type}")
    return unsupported_label, unsupported_label

def create_slider(name, params):
    container = QWidget()
    layout = QHBoxLayout(container) # Changed to QHBoxLayout
    layout.setContentsMargins(5, 2, 5, 2)
    
    label = QLabel(f"{name}:")
    label.setToolTip(params.get('tooltip', ''))
    
    value_label = QLabel(f"{params['value']}") # Separate label for the value
    
    line_edit = QLineEdit(str(params['value']))
    line_edit.setFixedWidth(50)

    if params['type'] == 'float':
        multiplier = 100
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(params['range'][0] * multiplier), int(params['range'][1] * multiplier))
        slider.setValue(int(params['value'] * multiplier))

        line_edit.setValidator(QDoubleValidator(params['range'][0], params['range'][1], 2))

        def update_from_slider(value):
            float_val = value / multiplier
            line_edit.setText(f"{float_val:.2f}")

        def update_from_line_edit():
            try:
                val = float(line_edit.text())
                slider.setValue(int(val * multiplier))
            except ValueError:
                pass # Ignore invalid input

        slider.valueChanged.connect(update_from_slider)
        line_edit.editingFinished.connect(update_from_line_edit)

    else: # int
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(params['range'][0], params['range'][1])
        slider.setValue(params['value'])

        line_edit.setValidator(QIntValidator(params['range'][0], params['range'][1]))

        def update_from_slider(value):
            line_edit.setText(str(value))

        def update_from_line_edit():
            try:
                slider.setValue(int(line_edit.text()))
            except ValueError:
                pass

        slider.valueChanged.connect(update_from_slider)
        line_edit.editingFinished.connect(update_from_line_edit)

    layout.addWidget(label)
    layout.addWidget(slider)
    layout.addWidget(line_edit)
    return container, {"slider": slider, "line_edit": line_edit, "label": label}

def create_checkbox(name, params):
    # For checkboxes, we'll create a container and a separate label to allow styling
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(5, 2, 5, 2)
    
    label = QLabel(name)
    label.setToolTip(params.get('tooltip', ''))
    
    checkbox = QCheckBox()
    checkbox.setChecked(params['value'])
    
    layout.addWidget(label)
    layout.addStretch()
    layout.addWidget(checkbox)
    
    return container, {"checkbox": checkbox, "label": label}

def create_reward_slider(name, params, base_value, is_team_specific=False):
    container = QWidget()
    
    # --- Checkbox (Name & Enable/Disable) ---
    checkbox = QCheckBox(name)
    checkbox.setChecked(params.get('enabled', True))
    checkbox.setToolTip(params.get('tooltip', ''))

    # --- Initial Values ---
    initial_multiplier = params['value']
    initial_final_value = (base_value * initial_multiplier) if base_value is not None else initial_multiplier

    # --- Line Edit (for Final Value) ---
    line_edit = QLineEdit(f"{initial_final_value:.2f}")
    line_edit.setFixedWidth(60) # A bit wider for larger numbers
    if base_value is not None:
        min_val, max_val = params['range'][0] * base_value, params['range'][1] * base_value
        if min_val > max_val: min_val, max_val = max_val, min_val # Handle negative base
        line_edit.setValidator(QDoubleValidator(min_val, max_val, 2))
    else: # Fallback
        line_edit.setValidator(QDoubleValidator(params['range'][0], params['range'][1], 2))

    # --- Slider (for Multiplier) ---
    slider = QSlider(Qt.Orientation.Horizontal)
    slider_precision = 100
    slider.setRange(int(params['range'][0] * slider_precision), int(params['range'][1] * slider_precision))
    slider.setValue(int(initial_multiplier * slider_precision))

    # --- Multiplier Label (for clarity) ---
    multiplier_label = QLabel(f"x{initial_multiplier:.2f}")
    multiplier_label.setFixedWidth(45)
    multiplier_label.setStyleSheet("color: #a9a9a9;")

    # --- Syncing Logic ---
    def update_controls_from_slider(slider_value):
        current_multiplier = slider_value / slider_precision
        multiplier_label.setText(f"x{current_multiplier:.2f}")
        if base_value is not None:
            final_value = base_value * current_multiplier
            # Block signals on line_edit to prevent feedback loop
            line_edit.blockSignals(True)
            line_edit.setText(f"{final_value:.2f}")
            line_edit.blockSignals(False)

    def update_slider_from_line_edit():
        try:
            final_val = float(line_edit.text())
            if base_value is not None and abs(base_value) > 1e-6:
                multiplier_val = final_val / base_value
                # Block signals on slider to prevent feedback loop
                slider.blockSignals(True)
                slider.setValue(int(multiplier_val * slider_precision))
                slider.blockSignals(False)
                # Also update the multiplier label since slider valueChanged won't fire
                multiplier_label.setText(f"x{multiplier_val:.2f}")
            elif abs(base_value) <= 1e-6: # Handle base value of 0
                slider.blockSignals(True)
                slider.setValue(0)
                slider.blockSignals(False)
                multiplier_label.setText(f"x0.00")
        except ValueError:
            pass

    slider.valueChanged.connect(update_controls_from_slider)
    line_edit.editingFinished.connect(update_slider_from_line_edit)

    # --- Enable/Disable Logic ---
    checkbox.toggled.connect(slider.setEnabled)
    checkbox.toggled.connect(line_edit.setEnabled)
    checkbox.toggled.connect(multiplier_label.setEnabled)
    slider.setEnabled(checkbox.isChecked())
    line_edit.setEnabled(checkbox.isChecked())
    multiplier_label.setEnabled(checkbox.isChecked())

    # --- Layout ---
    if is_team_specific:
        # Vertical layout for the cramped team settings view
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(5, 2, 5, 2)
        main_layout.setSpacing(2)
        
        top_container = QWidget()
        top_layout = QHBoxLayout(top_container)
        top_layout.setContentsMargins(0,0,0,0)
        
        top_layout.addWidget(checkbox)
        top_layout.addStretch()
        top_layout.addWidget(multiplier_label)
        top_layout.addWidget(line_edit)

        bottom_container = QWidget()
        bottom_layout = QHBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(15,0,0,0) # Indent the slider slightly
        bottom_layout.addWidget(slider)
        
        main_layout.addWidget(top_container)
        main_layout.addWidget(bottom_container)
    else:
        # Default horizontal layout
        layout = QHBoxLayout(container)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.addWidget(checkbox, 2)
        layout.addWidget(slider, 3)
        layout.addWidget(multiplier_label)
        layout.addWidget(line_edit, 1)

    return container, {"checkbox": checkbox, "slider": slider, "line_edit": line_edit, "label": checkbox}

def create_randomizable_slider(name, params):
    container = QWidget()
    main_layout = QVBoxLayout(container)
    main_layout.setContentsMargins(5, 2, 5, 2)
    main_layout.setSpacing(2)

    # --- Base Value Controls ---
    base_container = QWidget()
    base_layout = QHBoxLayout(base_container)
    base_layout.setContentsMargins(0,0,0,0)
    
    base_label = QLabel(f"{name}:")
    base_line_edit = QLineEdit(str(params['value']['base']))
    base_line_edit.setFixedWidth(50)
    
    base_slider = QSlider(Qt.Orientation.Horizontal)

    base_layout.addWidget(base_label)
    base_layout.addWidget(base_slider)
    base_layout.addWidget(base_line_edit)
    
    # --- Randomization Controls ---
    rand_container = QWidget()
    rand_layout = QHBoxLayout(rand_container)
    rand_layout.setContentsMargins(0,0,0,0)

    rand_label = QLabel("  └─ Rand (%):")
    rand_line_edit = QLineEdit(str(int(params['value']['rand'] * 100)))
    rand_line_edit.setValidator(QIntValidator(0, 100))
    rand_line_edit.setFixedWidth(50)

    rand_slider = QSlider(Qt.Orientation.Horizontal)
    rand_slider.setRange(0, 100)
    rand_slider.setValue(int(params['value']['rand'] * 100))

    rand_layout.addWidget(rand_label)
    rand_layout.addWidget(rand_slider)
    rand_layout.addWidget(rand_line_edit)

    # --- Add to main layout ---
    main_layout.addWidget(base_container)
    main_layout.addWidget(rand_container)

    # --- Syncing Logic ---
    if params['value_type'] == 'float':
        multiplier = 100
        base_slider.setRange(int(params['range'][0] * multiplier), int(params['range'][1] * multiplier))
        base_slider.setValue(int(params['value']['base'] * multiplier))
        base_line_edit.setValidator(QDoubleValidator(params['range'][0], params['range'][1], 2))
        base_line_edit.setText(str(params['value']['base']))


        def sync_base_slider_to_text():
            try: base_slider.setValue(int(float(base_line_edit.text()) * multiplier))
            except ValueError: pass
        def sync_base_text_to_slider():
            base_line_edit.setText(f"{base_slider.value() / multiplier:.2f}")

        base_slider.valueChanged.connect(sync_base_text_to_slider)
        base_line_edit.editingFinished.connect(sync_base_slider_to_text)
        sync_base_text_to_slider()
    else: # int
        base_slider.setRange(params['range'][0], params['range'][1])
        base_slider.setValue(params['value']['base'])
        base_line_edit.setValidator(QIntValidator(params['range'][0], params['range'][1]))
        base_line_edit.setText(str(params['value']['base']))


        def sync_base_slider_to_text():
            try: base_slider.setValue(int(base_line_edit.text()))
            except ValueError: pass
        def sync_base_text_to_slider():
            base_line_edit.setText(str(base_slider.value()))

        base_slider.valueChanged.connect(sync_base_text_to_slider)
        base_line_edit.editingFinished.connect(sync_base_slider_to_text)
        sync_base_text_to_slider()
    
    rand_val = params['value'].get('rand', 0.0)
    rand_line_edit.setText(str(int(rand_val * 100)))
    rand_slider.setValue(int(rand_val * 100))

    def sync_rand_slider_to_text():
        try: rand_slider.setValue(int(rand_line_edit.text()))
        except ValueError: pass
    def sync_rand_text_to_slider():
        rand_line_edit.setText(str(rand_slider.value()))
    
    rand_slider.valueChanged.connect(sync_rand_text_to_slider)
    rand_line_edit.editingFinished.connect(sync_rand_slider_to_text)

    return container, {"base_slider": base_slider, "base_line_edit": base_line_edit, "rand_slider": rand_slider, "rand_line_edit": rand_line_edit, "label": base_label}
