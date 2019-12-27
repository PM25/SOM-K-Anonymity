import time

def timeit(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return (result, end_time - start_time)

def profile(runners, X_train, y_train, X_test, y_test):
    runners_profile = {}
    for runner in runners:
        result, preprocess_time = timeit(runner.preprocess, X_train, y_train)
        X_preprocessed, y_preprocessed = result
        runner.fit(X_preprocessed, y_preprocessed)
        report = runner.evaluate(X_test, y_test)
        runner_profile = {
            **report,
            'preprocessed_time': preprocess_time
        }
        runners_profile[str(runner)] = runner_profile
    return runners_profile