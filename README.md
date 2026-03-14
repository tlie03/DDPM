# MNIST-DDPM

### TODO
- [x] change input shape in Denoise class, so that different shapes are allowed and the subclass can give tighter constraints
- [x] make sure that t ranges from (0, T-1) is consistent across all functions and classes
- [ ] add seeds
- [ ] add the right norm for DnnDenoise
- [x] change time encoding to positional -> embedding
- [x] make code cleaner (seperate functions for alpha, alpha_bar calculation)
- [ ] write tests, to ensure that training and test algorithm are completely correct
- [ ] make sampling algorithm a class
- [ ] reimplement training
- [ ] reimplement sampling




