

def optim_step(optimizer, optim_type, step, i):
    if ("extra" in optim_type.lower() and (step % 2 == 0 or i == 0)):
        optimizer.extrapolation()
    else:
        optimizer.step()

    return optimizer
