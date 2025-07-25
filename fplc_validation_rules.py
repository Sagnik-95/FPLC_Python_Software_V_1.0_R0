class MethodValidationEngine:
    def __init__(self):
        self.rules = {
            'max_pressure': lambda x: x <= 25.0,
            'max_flow_rate': lambda x: x <= 5.0,
            'gradient_limits': lambda x: 0 <= x <= 100,
            'max_duration': lambda x: x <= 999
        }
    
    def validate_method(self, method_steps):
        validation_results = []
        for step in method_steps:
            for rule_name, rule_func in self.rules.items():
                if not rule_func(step.get_parameter(rule_name)):
                    validation_results.append(f"Rule violation: {rule_name}")
        return validation_results
