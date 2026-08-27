"""Offline fixture smoke runner for the template task project."""

from praxist.core.task_project import TaskProject
from praxist.testing.fake_workflow_fixture import FakeWorkflowFixtureTaskRunner


def create_task_runner(task_project: TaskProject) -> FakeWorkflowFixtureTaskRunner:
    """Task project entrypoint that constructs the template fixture runner."""
    return FakeWorkflowFixtureTaskRunner(task_project)
