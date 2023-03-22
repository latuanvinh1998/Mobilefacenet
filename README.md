# Cheapfakes Detection Grand Challenge at ICME'23

## Description:

Previous proposal have utilize/mining the semantic/correlation between text and visual features in many aspect and achieve good performance.

However, beside semantic understanding of image and text, the external knowledge is very important. In our proposal, we inject knowledge understanding from knowledge graph to enhance the performance of COSMOS baseline on task 1 and achieve greate performance on task 2.


| Method  | Task 1 | Task 2 |
| ------------- | ------------- | ------------- |
| COSMOS  | x | x |
| [^1]  | x  | 76 |
| [^2]  | x  | 73 |
| Our proposal  | x  | 84 |


-  **Pull docker: `submission` from this repository and `bottom_up_attention from my previous work at ACMMM`:**

    ```sh
    docker pull latuanvinh1998/icmecheapfakes:submission
    docker pull latuanvinh1998/acmmmcheapfakes:bottom_up_attention
    ```
## Task 1.
- **Evaluate (2 examples):**
    ```sh
    docker run -v "path to folder containing the hidden test split file test.json":/icmecheapfakes --gpus all latuanvinh1998/icmecheapfakes:submission python eval_task_1.py > "outputfile"
    ```

    ```sh
    docker run -v path/to/folder/containing/test.json:/icmecheapfakes --gpus all latuanvinh1998/icmecheapfakes:submission python eval_task_1.py > outputfile.txt
    ```

 **NOTE:** *Base on description of https://www.2023.ieeeicme.org/author-info.php, we assume the json of test dataset is [test.json]. We also assume images folder [text] and [test.json]. is in same folder*

## Task 2.

 **NOTE:** *To evaluate task 2, our proposal need to run 2 step: Features Extract to extract features of image first, and Evaluate the proposal.*

- **Feature Extract (2 examples)::**
    ```sh
    docker run -v "path/to/folder containing the hidden test split file task_2.json":/acmmmcheapfakes --gpus all latuanvinh1998/acmmmcheapfakes:bottom_up_attention python extract_task_2.py
    ```

    ```sh
    docker run -v path/to/folder/containing/task_2.json:/acmmmcheapfakes --gpus all latuanvinh1998/acmmmcheapfakes:bottom_up_attention python extract_task_2.py
    ```

*After this command, docker will create `task_2.npy` and save at `path/to/folder/containing/task_2.json`.*

- **Evaluate (2 examples)::**
    ```sh
    docker run -v "path to folder containing the hidden test split file task_2.json":/icmecheapfakes --gpus all latuanvinh1998/icmecheapfakes:submission python eval_task_2.py > "outputfile"
    ```

    ```sh
    docker run -v path/to/folder/containing/task_2.json:/icmecheapfakes --gpus all latuanvinh1998/icmecheapfakes:submission python eval_task_2.py > outputfile.txt
    ```

 **NOTE:** *We assume the json file of task 2 is [task_2.json] and the images folder is [images_task_2]. We also assume [images_task_2] and [task_2.json] is in same folder*

[^1]: A Combination of Visual-Semantic Reasoning and Text Entailment-based Boosting Algorithm for Cheapfake Detection.
[^2]: A Textual-Visual-Entailment-based Unsupervised Algorithm for Cheapfake Detection.
