# Flip-S
Example code for Flip-S. We will release all code when the work is accepted.

# Set-up
Download the [Imagenet-1k](https://image-net.org/challenges/LSVRC/2012/index.php) dataset and replace the DataLoaderArguments.valdir parameter with the path of the valid dataset.

# Targeted Attack
Here we provide a example code and trigger to attack [DeiT-base](https://huggingface.co/facebook/deit-base-distilled-patch16-224) model. Just run the following command after you completing set-up.

```
python deit-targeted.py
```

# Untargeted Attack
In addition to the targeted attack, we also provide an example of the untargeted attack, which also targets the DeiT-base model. Please run the following command after completing the set-up for the attack.

```
python deit-untargeted.py
```
