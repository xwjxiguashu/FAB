from .resource_calendar import CalendarDecoderMixin
from .local_search import LocalSearchMixin
from .nsga2 import NSGAIIMixin
from .objectives import ObjectiveMixin
from .plotting import PlottingMixin
from .problem import ProblemDefinitionMixin
from .rolling import RollingSchedulerMixin


class TwoPopulationScheduler(
    ProblemDefinitionMixin,
    CalendarDecoderMixin,
    ObjectiveMixin,
    NSGAIIMixin,
    LocalSearchMixin,
    RollingSchedulerMixin,
    PlottingMixin,
):
    """FAB scheduler using two coevolving policy populations."""
    pass

