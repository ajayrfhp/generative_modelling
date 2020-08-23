# Goal of lecture
- Fit density model, $p_\theta(x)$ with continous $x$
- Desired properties of the model
  - Good fit to underlying data
  - Be able to predict $p_\theta(x)$ for a unknown x
  - Sample from the distribution 
  - Bonus goal 
    - Provide a useful latent representation

# Topics to cover
- Foundations of 1D flow
- 2-D flows
- N-D flows
- Dequantization

## Foundations of 1D flow
- Fit density models using minimizing MLE
- Does mixture of gaussian work well in higher dimensions ?
  - No. Why ?
    - Picking a random cluster and adding noise does not result in smooth images
     ![Fitting flow density model](./resources/fitting_flow_density_model.png)
   - Problem with continous flow fitting
     - We cannot just put a softmax to normalize over all possible states like we did in the discrete case
     - Without normalization, MLE will just push probability scores to infinity as thats the most optimal parameter values. With normalizer, you try to put mass on right areas and give low mass on wrong areas. 
 - **Main concept behind flows**
   - ![Flow main idea](./resources/flow_main_idea.png)
   - Flow is mapping from x to z, if z is normal then its called normalizing flow. 
  - How to use P(z) to estimate P(x) ?
  - Change of variable magic
    - ![Change of variable](./resources/change_of_variable.png) 