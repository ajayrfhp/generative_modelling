Models like pix2pix can translate from one domain to another. Existing models are inefficient at multi modal translations, they need kC2 generators, also they do not share information across the modes. 

StarGAN learns mapping across multiple domains using one generator. 

StarGAN can learn and transfer multi modal information across datasets. 

StarGAN provides superior results to compared models. 

Objective function
    Domain classification loss, adversarial loss and cycle consistency loss. 

Datasets
* CELEBA - 40 facial attributes for each image
* RAFD - 8 emotional attributes for each image

Baseline methods 
    - DIAT
    - CycleGAN
    - IcGAN

Evaluation
- Qualitative
- Quantitative
    - Amazon mechanical turks and transfer learning classification accuracy
    - Star GAN wins big in multi attribute cases. 
    - StarGAN can recognize facial expressions better than baseline. 
    - StarGAn has fewer parameters than baseline

Conclusion
- Stargan can scale across domains, datasets and produce better results than baselines compared in the dimensions compared. 
