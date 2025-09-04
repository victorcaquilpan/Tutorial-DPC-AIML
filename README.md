## DPC Demo 

Short demo/tutorial how you can use the AIML' shared computing resource, known as **Deep Puple Cluster (DPC)**. This is a different cluster than Phoenix and it can be used for any AIML member. Unlike Phoenix, DPC uses Kubernetes and Docker containers for deployment. 

Some advantages of using DPC:

* GPU availability (see the GPUs available below)
* You can run many experiments (it depends on availability)
* Option of running in multiple GPUs 
* Max. default storage of 1TB (you can ask Hui to get more temporal storage)
* Internet connection (great for recording the logs in real time - try [Wandb](https://wandb.ai/) 😀)

Cluster Composition

```
  1 x Ada A6000 Node
  1 x L40S Node
  8 x A100 Nodes
  3 x DGX Nodes (with V100 GPUs)
     - 1 x DGX-1 (8 GPUs)
     - 2 x DGX-2 (16 GPUs each)
```

There is already a [main tutorial](https://github.com/aiml-au) for DPC use, however, the current demo is created in a higher level (dummy demo).

### Steps 

First, clone this repository by running:

```
git clone https://github.com/victorcaquilpan/Tutorial-DPC-AIML.git
cd Tutorial-DPC-AIML
```

You need an @aiml.team account for accesisng to DPC. So, the first step is sending an email to admins@aiml.team requesting for the use of DPC. You need to CC your AIML supervisor. 

Once, you get your @aiml.team account, you can access to the main DPC documentation here: https://help.cluster.aiml.team/. You can follow each one of the sections, however, since it might be a bit overwhelming, I leave here main steps for an easy use. First, you need to follow all the steps indicated in the **Preparation** section.

We will be running a basic image classification model for [fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist). I created a basic script in python, then we would go to run it in a Docker container inside DPC.

1) **Create PVC on DPC**. You can use the file `dpc-files/pvc`, where we are defining the storage for our project (data, results, etc). We are using 2Gib for this project. The maximum is 1000Gib. **NOTE**: You can ask the AIML's System Adm for more storage if you need. Run: 

```
cd dpc-files
kubectl create -f pvc.yaml
```

For your own project, change the pvc name  (e.g. my-project_pvc).

2) **Transferring data to Pods** (smallest deployable unit of computing in Kubernetes): Use the file `dpc-files/data-transfering.yaml` and run: 

```
kubectl create -f data-transfering.yaml
```

Here, you need to change the metadata name (e.g. my-data_transfer) and match the PVC container name defined in the previous step (**data-transfer-pvc**).

After doing this, you can check a new Pod (smallest deployable unit in Kubernetes) is created, running:
```
kubectl get pods
```
OUTPUT:
```
NAME                                     READY   STATUS    RESTARTS   AGE
mnist-data-transfer-bx9qp   1/1     Running   0          76s
```
This means, a docker container is running, which is responsible to handle your data. For each Pod, an random text is added to the end of the name pod  (in this case **bx9qp**). This text is useful as an identifier for Kubernetes. Now, you can transfer your data from your local workstation to the data Pod, by running:
```
kubectl cp ./../data/ mnist-data-transfer-bx9qp:/data/
```
You can check if your data was transfered successfully by accessing to the Pod running: 
```
kubectl exec -it mnist-data-transfer-bx9qp  -- /bin/bash
cd /data
ls
```
OUTPUT:
```
data/  lost+found/ mnist-data/
```

**mnist-data** corresponds to your data.

3) Creating training Pods

Now, knowing your data is ready to use, you can create multiple YAML files for different experiments. For the experiments, you need to define mainly:

* The docker image where you want to run your scripts. There are some internal images in docker.aiml.team. Otherwise, you can use any public docker image.  
* Get the scripts. You can get the scrips by cloning a GitHub/GitLab repository. Also the script can be inside the image. 

Inside the YAML file, you can define all the bash scripts that you need to run inside the Docker container. Also, you can add arguments to your script. For this example, we are using this repository to get the script to run.

```
cd training-jobs
kubectl create -f experiment1.yaml
```




