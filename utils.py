import torch
import matplotlib.pyplot as plt


def norm(vec):
    return torch.nn.functional.normalize(vec, p=2, dim=1)


def mag(vec):
    return vec.norm(p=2, dim=1, keepdim=True)


def plot(points, trainer, epoch):
    figure, axes = plt.subplots()
    uc_3 = plt.Circle(trainer.c, trainer.R, fill=False)

    new_x = []
    new_y = []

    for data in points:
        if len(data) == 50:
            mapped = trainer.model.forward(norm(data)).cpu().detach().numpy()
        else:
            mapped = trainer.model.forward(torch.nn.functional.normalize(data, p=2, dim=0)).cpu().detach().numpy()

        if mapped.shape == (50, 2):
            for m in mapped:
                new_x.append(m[0])
                new_y.append(m[1])
        else:
            new_x.append(mapped[0])
            new_y.append(mapped[1])

    plt.scatter(new_x, new_y,
                color="red", s=5)
    plt.scatter(trainer.c.cpu().detach().numpy()[0], trainer.c.cpu().detach().numpy()[1],
                color="black", s=5)
    axes.add_artist(uc_3)
    plt.gca().add_patch(uc_3)
    plt.axis('equal')
    plt.savefig(f'state/state-{epoch}.png')
    plt.close(figure)
