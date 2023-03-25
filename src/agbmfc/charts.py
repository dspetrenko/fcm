import matplotlib.pyplot as plt


def plot_target_and_pred(target, pred):
    plt.figure()

    f, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 10))
    axes[0].imshow(target, vmin=0, vmax=255, interpolation='nearest')
    axes[0].title.set_text('target')
    axes[1].imshow(pred, vmin=0, vmax=255, interpolation='nearest')
    axes[1].title.set_text('prediction')

    plt.show()
