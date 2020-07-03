## Mobilenet

### Key Contributions
- Build compute efficient neural networks for low compute devices by using depth separable convolutions.
- Provides 2 hyperparameters to trade off accuracy and inference latency

### Mobilenet architecture

- **Question** How many parameters does a conv net that maps from `(w, h, in)` to `(w, h, out)` have ? What is the associated computational cost ?

  - Conv Net - (num filters `out`, kernel size `k`, padding same)
  - $Parameters = k \cdot k \cdot out \cdot in$
  - $Cost = k \cdot k \cdot w \cdot h \cdot out  \cdot in $ 

- **Depthwise separable convolution**
  - Break a 3 * 3 convolution into 3*1 and 1 * 3
  - $Cost = k \cdot k \cdot w \cdot h \cdot out + w \cdot h \cdot out \cdot in$
  

  - **Hyper parameters**
    - Width reducer $\alpha$ controls number of output channels based on input. 
      - $out = \alpha \cdot in$
      - $Cost= k \cdot k \cdot w \cdot h \cdot \alpha \cdot in + w \cdot h \cdot \alpha \cdot in \cdot in$
    - Resolution Multiplier $p$ controls image resolution
      - Size of output image is ($pw$, $ph$, $\alpha\cdot in$) 
      - $Cost= k \cdot k \cdot pw \cdot ph \cdot \alpha \cdot in + w \cdot ph \cdot \alpha \cdot in \cdot in$

-  **Efficiency**

   - $Efficiency = \frac{Parameters\:Mobilenet}{Parameters\:ConvNet}$ 
   - $Efficiency = \frac{1}{in} + \frac{1}{k^2}$ 
   - Mobilenet for $k=3$ uses 8-9 times less computations


- **Takeaways**
  - Breakdown convolutions into depthwise and pointwise convolutions to factorize computational cost. 
  - $\alpha$ to control trade off between accuracy and latency

- **Resources**
  - [Mobilenet paper](https://arxiv.org/abs/1704.04861)
  - [Mobilenet video](https://www.youtube.com/watch?v=HD9FnjVwU8g)