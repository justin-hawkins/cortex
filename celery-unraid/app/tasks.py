"""
Example Celery tasks.
Replace or extend these with your actual application tasks.
"""
from app.celery_app import app


@app.task(bind=True)
def add(self, x, y):
    """Add two numbers together."""
    return x + y


@app.task(bind=True)
def multiply(self, x, y):
    """Multiply two numbers together."""
    return x * y


@app.task(bind=True)
def hello(self, name='World'):
    """A simple greeting task."""
    return f'Hello, {name}!'


@app.task(bind=True)
def long_running_task(self, seconds=10):
    """
    A task that simulates long-running work.
    Updates task state with progress information.
    """
    import time
    
    for i in range(seconds):
        time.sleep(1)
        self.update_state(
            state='PROGRESS',
            meta={'current': i + 1, 'total': seconds}
        )
    
    return {'status': 'completed', 'result': f'Finished after {seconds} seconds'}