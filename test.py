import torch
import visdom
def main():
    vis = visdom.Visdom()
    vis.line([0], [-1], win='jdsklfdfhdjkfhjdk', opts=dict(title='jdsklfdfhdjkfhjdk'))


if __name__ == '__main__':
    main()