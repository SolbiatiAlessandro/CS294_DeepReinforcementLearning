import logging
import requests

def parse_job_dir(cloud_job_dir):
    """
    cloud_job_dir:
    "relna_imitation_learning__31__fbe799c5_6504_4b35_b104_af46a6a11203"

    return:
    31
    """
    try:
        # parse job request
        jobdir_to_parse = cloud_job_dir
        start = jobdir_to_parse.find("__")
        end = jobdir_to_parse[start + 1:].find("__")
        jobid = jobdir_to_parse[start+2:start+1+end]
        logging.warning("utils.parse_job_dir: succesfully parsed {} -> {}".format(
            cloud_job_dir,
            jobid
            ))
        return int(jobid)
    except Exception as e:
        print(e)
        logging.error("PARSE ERROR OF JOB DIR: {}".format(cloud_job_dir))
        logging.error("UPDATING JOB WITH ID 1 INSTEAD OF QUITTING JOB")
        return 1

def post_model_outputs(
        jobid,
        output_1,
        output_2,
        table_name="imitation_learning_jobs",
        url='https://relna-241818.appspot.com/submit_job_results'):
    """
    """
    values = {
            "table_name":table_name,
            "jobid":jobid,
            "output_1":output_1,
            "output_2":output_2
            }
    logging.warning("task: opening connectino with {}".format(url))
    logging.warning("task: posting request to server {}".format(values.values()))
    response = requests.post(url, values)
    logging.warning("task: request successful")
    return response

