The following are equivalent

    > dlnorm(x, 2, 0.5)
    > stats.lognorm.pdf(scale=np.exp(2), s=0.5, x)

    > dgamma(x, 2, 1/2)
    > stats.gamma.pdf(a=2, scale=1/2, x)