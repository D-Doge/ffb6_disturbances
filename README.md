# Introducing a novel approach to analyse 6D Pose estimators under disturbances


## Prerequisites

Before you begin, ensure you have the following installed on your machine:

- [Docker](https://www.docker.com/get-started)
- Download the [YCB-Video dataset](https://rse-lab.cs.washington.edu/projects/posecnn/)
- Download the [pretrained weights](https://github.com/ethnhe/FFB6D) and move them into the YCB-Video dataset directory, insinde the subdirectory called /ffb6d_checkpoints.

## Usage

Build the Dockerfile inside /docker.

```
docker build -t ma_ffb6d .
```
Run the docker container, mounting the YCB-Video dataset, with the pretrained weights inside.
```
docker run -it -v E:\Uni\MA\ss24_niedermt_analysis_6d_pose_estimation\docker:/workspace/dd6d -v PATH_TO_YCB_VIDEO_DATASET:/workspace/dd6d/datasets/ycb/YCB_Video_Dataset --gpus all --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 --name ma_6d_pose ma_ffb6d
```
Adjust distrubances in docker\test_model.py and run the file to get the metrics and scores for the distrubance.

## License
```markdown
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

---

Feel free to modify the template to fit the specific requirements and structure of your project.