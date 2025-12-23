"""
Configuration Management

This module provides utilities for loading and validating analysis configurations
from YAML/TOML files.

Key Features:
- YAML/TOML configuration files
- Default parameter management
- Validation
- Easy parameter access

Author: Mykyta Bobylyow
Date: 2025
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import warnings


class AnalysisConfig:
    """Configuration container for protein orientation analysis."""

    # Default parameters
    DEFAULTS = {
        # Trajectory settings
        'trajectory': {
            'start': 0,
            'stop': None,
            'step': 1,
            'selection': 'protein',
            'center': True,
            'align': False
        },
        # Analysis settings
        'analysis': {
            'temperature': 300.0,  # Kelvin
            'pmf_bins': [36, 18, 36],  # phi, theta, psi
            'acf_max_lag': None,  # Auto: n_frames // 4
            'diffusion_fit_range': None  # Auto
        },
        # Output settings
        'output': {
            'directory': 'analysis_results',
            'save_euler': True,
            'save_pmf': True,
            'save_plots': True,
            'format': 'npz'  # 'npz' or 'hdf5'
        },
        # Visualization settings
        'visualization': {
            'dpi': 300,
            'figsize': [10, 8],
            'pmf_vmax': 10.0,  # kcal/mol
            'colormap': 'viridis'
        }
    }

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.

        Args:
            config_dict: Optional dictionary with user settings.
                        If None, uses defaults.
        """
        # Start with defaults
        self._config = self._deep_copy(self.DEFAULTS)

        # Update with user settings
        if config_dict is not None:
            self._update_recursive(self._config, config_dict)

    @staticmethod
    def _deep_copy(d: Dict) -> Dict:
        """Deep copy dictionary."""
        import copy
        return copy.deepcopy(d)

    @staticmethod
    def _update_recursive(base: Dict, update: Dict) -> None:
        """Recursively update nested dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                AnalysisConfig._update_recursive(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'trajectory.start')
            default: Default value if key not found

        Returns:
            value: Configuration value

        Example:
            >>> config.get('analysis.temperature')
            300.0
        """
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Args:
            key: Configuration key
            value: New value

        Example:
            >>> config.set('analysis.temperature', 310.0)
        """
        keys = key.split('.')
        d = self._config

        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]

        d[keys[-1]] = value

    def to_dict(self) -> Dict:
        """Export configuration as dictionary."""
        return self._deep_copy(self._config)

    def save(self, filename: str) -> None:
        """
        Save configuration to file.

        Supports .json, .yaml, .toml formats based on extension.

        Args:
            filename: Output file path

        Example:
            >>> config.save('analysis_config.yaml')
        """
        path = Path(filename)
        suffix = path.suffix.lower()

        if suffix == '.json':
            with open(filename, 'w') as f:
                json.dump(self._config, f, indent=2)

        elif suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(filename, 'w') as f:
                    yaml.dump(self._config, f, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML required. Install with: pip install pyyaml")

        elif suffix == '.toml':
            try:
                import toml
                with open(filename, 'w') as f:
                    toml.dump(self._config, f)
            except ImportError:
                raise ImportError("toml required. Install with: pip install toml")

        else:
            raise ValueError(f"Unsupported format: {suffix}. Use .json, .yaml, or .toml")

        print(f"✓ Configuration saved to {filename}")

    @classmethod
    def load(cls, filename: str) -> 'AnalysisConfig':
        """
        Load configuration from file.

        Args:
            filename: Configuration file (.json, .yaml, .toml)

        Returns:
            config: AnalysisConfig instance

        Example:
            >>> config = AnalysisConfig.load('my_config.yaml')
        """
        path = Path(filename)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filename}")

        suffix = path.suffix.lower()

        if suffix == '.json':
            with open(filename, 'r') as f:
                config_dict = json.load(f)

        elif suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(filename, 'r') as f:
                    config_dict = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required. Install with: pip install pyyaml")

        elif suffix == '.toml':
            try:
                import toml
                with open(filename, 'r') as f:
                    config_dict = toml.load(f)
            except ImportError:
                raise ImportError("toml required. Install with: pip install toml")

        else:
            raise ValueError(f"Unsupported format: {suffix}")

        return cls(config_dict)

    def validate(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            valid: True if all parameters are valid

        Raises:
            ValueError: If invalid parameters found
        """
        # Temperature must be positive
        temp = self.get('analysis.temperature')
        if temp <= 0:
            raise ValueError(f"Temperature must be positive, got {temp}")

        # PMF bins must be positive integers
        bins = self.get('analysis.pmf_bins')
        if not all(isinstance(b, int) and b > 0 for b in bins):
            raise ValueError(f"PMF bins must be positive integers, got {bins}")

        # Output format must be valid
        fmt = self.get('output.format')
        if fmt not in ['npz', 'hdf5']:
            raise ValueError(f"Output format must be 'npz' or 'hdf5', got {fmt}")

        return True

    def __repr__(self) -> str:
        """String representation."""
        return f"AnalysisConfig({self._config})"

    def __str__(self) -> str:
        """Pretty print configuration."""
        import pprint
        return pprint.pformat(self._config, indent=2)


def create_default_config(filename: str = 'config.yaml') -> None:
    """
    Create default configuration file.

    Args:
        filename: Output file path

    Example:
        >>> create_default_config('my_analysis.yaml')
    """
    config = AnalysisConfig()
    config.save(filename)
    print(f"✓ Default configuration saved to {filename}")
    print("Edit this file to customize your analysis")


def load_config_with_overrides(config_file: Optional[str] = None,
                               **overrides) -> AnalysisConfig:
    """
    Load configuration with command-line overrides.

    Args:
        config_file: Optional configuration file path
        **overrides: Keyword arguments to override config values

    Returns:
        config: AnalysisConfig instance

    Example:
        >>> config = load_config_with_overrides(
        ...     'my_config.yaml',
        ...     temperature=310.0,
        ...     output_directory='results_310K'
        ... )
    """
    # Load from file or use defaults
    if config_file is not None:
        config = AnalysisConfig.load(config_file)
    else:
        config = AnalysisConfig()

    # Apply overrides
    for key, value in overrides.items():
        # Convert underscores to dots for nested keys
        key_dotted = key.replace('__', '.')
        config.set(key_dotted, value)

    # Validate
    config.validate()

    return config


if __name__ == '__main__':
    # Example usage
    print("Configuration Management Module")
    print("================================")
    print()
    print("Example usage:")
    print()
    print("from protein_orientation.config import AnalysisConfig, create_default_config")
    print()
    print("# Create default config file")
    print("create_default_config('analysis.yaml')")
    print()
    print("# Load and modify")
    print("config = AnalysisConfig.load('analysis.yaml')")
    print("config.set('analysis.temperature', 310.0)")
    print("config.set('output.directory', 'results_310K')")
    print()
    print("# Access parameters")
    print("temp = config.get('analysis.temperature')")
    print("print(f'Temperature: {temp} K')")
    print()
    print("# Validate")
    print("config.validate()")
