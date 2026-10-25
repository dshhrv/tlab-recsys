def start_task(name, config, enabled=True):
    if not enabled:
        return None
    from clearml import Task
    task = Task.init(project_name='RecSys RL Task 2', task_name=name, reuse_last_task_id=False)
    task.connect(config, name='Configuration', ignore_remote_overrides=True)
    return task
