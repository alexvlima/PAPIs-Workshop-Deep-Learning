# Building Deep Learning application with Keras: workshop outline

*Author: Felipe Salvatore, Lucas Moura © All Rights Reserved*

----------
## Introduction

In this hands-on workshop for beginners, we’ll go through the core concepts of Deep Learning, its practical application to real life problems, and we’ll see how it achieves state of the art results on Computer Vision and Natural Language Processing. 

## Check set-up
[ ] Wifi access

[ ] Access [tlk.io](http://tlk.io/ml-humancoders)/papis-dl (live chat)

### Floyd, Jupyter, Python

----------

⚠️ **Please don’t use the** ***Workspace*** **feature in Floyd**

----------
[ ] Have you installed `floyd-cli`?

[ ] Send your Floyd username to your instructor / to the chat room (we’ll add you to our Team on Floyd)

[ ] From your terminal, create a new directory on your machine, and link it to our team project:
- `mkdir papis-dl`
- `cd papis-dl`
- `floyd init humancoders/projects/papis-dl`
    
[ ] Add the following notebooks to your local `papis-dl` directory
- <INSERT_LINK>

[ ] Run Jupyter server on Floyd (using CPU):

- `floyd run --cpu --data lucasmoura/datasets/self-drive-data-papis-workshop/1:self_drive_data_papis_workshop --data lucasmoura/datasets/efigi-papis-2018/1:efigi_data --mode jupyter`

[ ] Run Jupyter server on Floyd (using GPU):

- `floyd run --gpu --data lucasmoura/datasets/self-drive-data-papis-workshop/1:self_drive_data_papis_workshop --data lucasmoura/datasets/efigi-papis-2018/1:efigi_data --mode jupyter`

Note that this uploads the contents of your local directory to Floyd.

[ ] Your run/job will appear at [https://www.floydhub.com/humancoders/projects/papis-dl](https://www.floydhub.com/humancoders/projects/papis-dl)

[ ] In the Jupyter server, click on `intro`

[ ] Go through notebook `intro.ipynb`

[ ] In the Jupyter server, click on `dfn_cnn`

[ ] Go through notebook `dfn_cnn.ipynb`

[ ] And if you want, go through notebook `transfer.ipynb`

[ ] In the Jupyter server, click on `dfn_cnn`

[ ] Go through notebook `lstm.ipynb`

[ ] And if you want, go through notebook `Advanced Practices for Recurrent Neural Networks.ipynb`

[ ] In the Jupyter server, click on `api`

[ ] Go through notebook `api.ipynb`

[ ] Stop the Floyd run/job: `floyd stop [INSERT_JOB_NAME]`

**Remarks**

- Floyd is not a code repository. It’s just an “execution environment”.
  - It just gives access to cloud instances, for a limited time.
  - As a by-product, it stores the code that was sent from your machine to the cloud instance when you did `floyd run`, and the final versions of files before you destroyed that instance.
- We get billed as soon as Floyd gives you a job name, until we stop the run/job.
- Code and notebook changes are not automatically downloaded to your machine (but they’re easy to get, even after the cloud instance was destroyed).
- You can use different kinds of GPU and CPU by just [changing the flag](https://docs.floydhub.com/guides/run_a_job/#instance-type) [to the run command](https://docs.floydhub.com/guides/run_a_job/#instance-type).

### Not using Floyd?

Just download the notebooks and run locally:

<INSERT_LINK>

## Workshop structure

**Theory**

All theory needed for this workshop can be found in these three presentations: 

https://www.dropbox.com/s/dlzlz2bgfqdksv5/intro.pdf

https://www.dropbox.com/s/jcjv7ttjp4jg6m2/dfn_cnn.pdf

https://www.dropbox.com/s/4ik7py8q4pr9vph/rnn.pdf

**Hands on**

In total there will be 5 activities

 - Introduction(`intro/intro.ipynb`)
 - Feedforward and convolutional networks(`dfn_cnn/dfn_cnn.ipynb`)
 - Transfer learning(`dfn_cnn/transfer.ipynb`)
 - Recurrent neural network(`lstm/intro_to_recurrent_neural_networks.ipynb`)
 - Advanced recurrent neural network(`lstm/advanced_practices_for_recurrent_neural_networks.ipynb`)
 - API(`api/api.ipynb`)

You should work through all the tutorials to have a good grasp on building an API using a deep learning.

## Resources to go further

We have prepared some material to indicate your next steps in the deep learning journey

https://www.dropbox.com/s/m21z0g015l2yfhh/next_steps.pdf

## Copyright

Felipe Salvatore, Lucas Moura  © All Rights Reserved

