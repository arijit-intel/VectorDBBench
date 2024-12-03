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







