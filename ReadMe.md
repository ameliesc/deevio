## Shallow CNN Algorithm for classifying Nail images into Good or Bad.

* Algorithm can be trained using

      `python train.py`

  saves the trained weights (h5) and a keras model (json).


* RestAPI can be found in `classificationAPI.py` and can be run and tested using the following commands:

    `python classificationAPI.py`
    `curl -X POST -F image=@nail.jpeg 'http://localhost:5000/predic'`


* Package requirments can be found in `requirements.txt` and can be installed using

    `source setup.sh`



