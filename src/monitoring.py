from prometheus_client import Summary, Counter


METRIC_STORAGE = {
    'created_inference_tasks': Counter('fcm_created_inference_tasks_total',
                                       'The total number of created inference tasks'),
    'requests_to_fetch_inference_result': Counter('fcm_requests_to_fetch_inference_tasks_total',
                                                  'The total number of created inference tasks'),
    'echo_request_duration': Summary(name='fcm_echo_request_processing_seconds',
                                     documentation='Time spent processing request'),

}
