"""
Plugin discovery and registration.

Central registry that discovers, loads, and manages plugins by type.

Usage:
    from tokenforge.plugins.registry import PluginRegistry

    # Register a custom scheduler
    PluginRegistry.register(MyScheduler)

    # Discover
    schedulers = PluginRegistry.list_plugins("scheduler")
    my_sched = PluginRegistry.get("scheduler", "my_scheduler")
"""

from typing import Optional


class PluginRegistry:
    """
    Global plugin registry.

    Stores plugins indexed by (type, name) and provides
    discovery and instantiation methods.
    """

    _plugins: dict[str, dict[str, type]] = {}

    @classmethod
    def register(cls, plugin_cls: type, name: Optional[str] = None):
        """
        Register a plugin class.

        Args:
            plugin_cls: The plugin class to register.
            name: Override name (default: plugin_cls.name).
        """
        plugin_type = getattr(plugin_cls, "plugin_type", "unknown")
        plugin_name = name or getattr(plugin_cls, "name", plugin_cls.__name__.lower())

        if plugin_type not in cls._plugins:
            cls._plugins[plugin_type] = {}

        cls._plugins[plugin_type][plugin_name] = plugin_cls

    @classmethod
    def get(cls, plugin_type: str, name: str, **kwargs):
        """
        Instantiate a plugin by type and name.

        Args:
            plugin_type: Plugin category (scheduler, cache, attention, quantizer).
            name: Plugin name.
            **kwargs: Constructor arguments.

        Returns:
            Instantiated plugin.
        """
        if plugin_type not in cls._plugins:
            raise ValueError(
                f"Unknown plugin type '{plugin_type}'. "
                f"Available: {list(cls._plugins.keys())}"
            )
        if name not in cls._plugins[plugin_type]:
            available = list(cls._plugins[plugin_type].keys())
            raise ValueError(
                f"Unknown {plugin_type} plugin '{name}'. "
                f"Available: {available}"
            )
        return cls._plugins[plugin_type][name](**kwargs)

    @classmethod
    def list_plugins(cls, plugin_type: Optional[str] = None) -> dict:
        """
        List registered plugins.

        Args:
            plugin_type: Filter by type. If None, returns all.

        Returns:
            Dict mapping names to plugin classes.
        """
        if plugin_type:
            return dict(cls._plugins.get(plugin_type, {}))
        return {
            ptype: dict(plugins)
            for ptype, plugins in cls._plugins.items()
        }

    @classmethod
    def list_types(cls) -> list[str]:
        """Return all registered plugin types."""
        return list(cls._plugins.keys())

    @classmethod
    def clear(cls):
        """Clear all registered plugins (useful for testing)."""
        cls._plugins.clear()

    @classmethod
    def summary(cls) -> str:
        """Human-readable summary of registered plugins."""
        lines = ["TokenForge Plugin Registry:"]
        for ptype, plugins in cls._plugins.items():
            lines.append(f"  {ptype}:")
            for name, pcls in plugins.items():
                desc = getattr(pcls, "description", "")
                lines.append(f"    - {name}: {desc}" if desc else f"    - {name}")
        return "\n".join(lines)
