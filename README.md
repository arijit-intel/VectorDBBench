# Installation
1.	Clone the git repository from [here](https://github.com/arijit-intel/VectorDBBench/tree/changes_for_vdms).
2.	Follow the installation instructions from [here](https://github.com/zilliztech/VectorDBBench).

# Start VDMS docker
1. docker pull intellabs/vdms:latest
2. docker run $DOCKER_PROXY_RUN_ARGS --rm -a stdout -a stderr -p 55555:55555 --name vdms_new intellabs/vdms:latest

# Launch VectorDBBench
1. Run init_bench

  ![image](https://github.com/user-attachments/assets/0643655b-0c29-4eb2-a690-563d6018a153)

2. Copy and paste the network URL(adding ‘/run_test’) to the web browser, it will look like below:

   ![image](https://github.com/user-attachments/assets/419596fd-8160-4d55-b804-0fa05f55068d)

3. Then checkmark the VDMS, it will prompt to input some fields like below:

   ![image](https://github.com/user-attachments/assets/aed893b3-4dcb-4cc2-ab54-1c5d1b75fc87)

4. Enter the details as below(db_label is the name of the collection, distance strategy can be L2, IP. Engine can be FaissFlat, FaissIVFFlat, Flinng etc.):

   ![image](https://github.com/user-attachments/assets/52589806-d254-49e3-b02b-9d13d8c24c06)

5. Select the test data for making the index and search:

   ![image](https://github.com/user-attachments/assets/fe8c9cc9-eaee-4bb3-bce1-fd571cf6f952)

6. Run it(check ‘index already exists’ if the index already created):

   ![image](https://github.com/user-attachments/assets/3ac8c3e7-93c0-4209-a02b-2e7b9061d328)

7. You can check the results for insert(I changed the code to insert 10K entries, subsequent instruction describes how to do so) and search time(I changed the code to search 10 entries for illustration purpose, default is 1K, subsequent instruction describes how to do so):

   ![image](https://github.com/user-attachments/assets/b92987f5-820a-4c59-923a-573585ddcda8)

# Few How-tos:

## How to change insert element size?
  - Go to the backend/runner/serial_runner.py, and change as below:

    For 10K insert,
    
    ![image](https://github.com/user-attachments/assets/b650f034-1586-4c6c-82e3-d872f3730a83)

    For 100K insert,

    ![image](https://github.com/user-attachments/assets/f0ca55b7-9211-4c51-8bc7-69e79031a39a)

    For 1M insert,

    ![image](https://github.com/user-attachments/assets/e2925b2f-57d9-4ffc-bd42-3f95af562be5)

## How to change number of searches:
  - Go to the backend/runner/serial_runner.py, and change as below (default is 1K, in the below example it is changed to 10):

    ![image](https://github.com/user-attachments/assets/9a8b9095-ee62-4a31-b678-01ac0e76280d)

## How to change the insert batch size:
  - Go to the backend/clients/vdms/vdms_batch.py, and change as below (Here it’s set to 512):

    ![image](https://github.com/user-attachments/assets/943ec764-3b27-426f-bb9f-84839dab6700)

## How to change the search K size:
  - Go to the backend/clients/vdms/vdms_batch.py, and change as below (Here it’s set to 10):

    ![image](https://github.com/user-attachments/assets/dc535621-af36-461c-888d-c5277a1e42c8)

## How to add constraint in search?
  - First add metadata while inserts. Like below(here for example we added id and date as metadata):

    ![image](https://github.com/user-attachments/assets/255b43d8-a485-4b63-a6a2-5e9e4b92ad6f)

  -	Pass the constraint for while searching. Like below(here we are searching for ids which should be in between 6000 and 7000):

    ![image](https://github.com/user-attachments/assets/6975a9fc-589d-4ffd-82c7-e7c6917633ea)














