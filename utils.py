import torch
import matplotlib.pyplot as plt


def norm(vec):
    return torch.nn.functional.normalize(vec, p=2, dim=1)


def mag(vec):
    return vec.norm(p=2, dim=1, keepdim=True)


def plot(validation_data, trainer, epoch):
    figure, axes = plt.subplots()
    uc_3 = plt.Circle(trainer.c, trainer.R, fill=False)

    ps_normal = list(filter(lambda x: x[1] == 'benign', validation_data))
    ps_normal = torch.stack(list(map(lambda x: x[0], ps_normal)))
    ps_anormal = list(filter(lambda x: x[1] == 'malware', validation_data))
    ps_anormal = torch.stack(list(map(lambda x: x[0], ps_anormal)))

    ps_normal = trainer.model.forward(ps_normal)
    ps_anormal = trainer.model.forward(ps_anormal)

    normal_x = []
    normal_y = []
    anormal_x = []
    anormal_y = []

    for p in ps_normal.cpu().detach().numpy():
        normal_x.append(p[0])
        normal_y.append(p[1])

    for p in ps_anormal.cpu().detach().numpy():
        anormal_x.append(p[0])
        anormal_y.append(p[1])

    plt.scatter(anormal_x, anormal_y, color="red", s=5)
    plt.scatter(normal_x, normal_y,
                color="green", s=5)
    plt.scatter(trainer.c.cpu().detach().numpy()[0], trainer.c.cpu().detach().numpy()[1],
                color="black", s=5)
    axes.add_artist(uc_3)
    plt.gca().add_patch(uc_3)
    plt.axis('equal')
    plt.savefig(f'state/state-{epoch}.png')
    plt.close(figure)


def plot_validation(points_normal, points_anormal, trainer):
    figure, axes = plt.subplots()
    uc_3 = plt.Circle(trainer.c, trainer.R, fill=False)

    normal_x = []
    normal_y = []
    anormal_x = []
    anormal_y = []

    normal = trainer.model.forward(torch.stack(points_normal))
    a_normal = trainer.model.forward(torch.stack(points_anormal))

    for p in normal.cpu().detach().numpy():
        normal_x.append(p[0])
        normal_y.append(p[1])

    for p in a_normal.cpu().detach().numpy():
        anormal_x.append(p[0])
        anormal_y.append(p[1])

    plt.scatter(anormal_x, anormal_y, color="red", s=5)
    plt.scatter(normal_x, normal_y,
                color="green", s=5)
    plt.scatter(trainer.c.cpu().detach().numpy()[0], trainer.c.cpu().detach().numpy()[1],
                color="black", s=5)
    axes.add_artist(uc_3)
    plt.gca().add_patch(uc_3)
    plt.axis('equal')
    plt.savefig('state/state-validation-final.png')
    plt.close(figure)
