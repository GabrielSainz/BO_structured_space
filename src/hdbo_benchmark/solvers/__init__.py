"""Local optimization methods, with lazy legacy class exports."""

__all__ = ["COWBOYSDiffusion", "LatentSpaceBayesianOptimization"]


def __getattr__(name):
    if name == "COWBOYSDiffusion":
        from .cowboys_diffusion import COWBOYSDiffusion

        return COWBOYSDiffusion
    if name == "LatentSpaceBayesianOptimization":
        from .latent_space_bo import LatentSpaceBayesianOptimization

        return LatentSpaceBayesianOptimization
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
