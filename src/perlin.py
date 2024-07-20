# ported from https://github.com/pvigier/perlin-numpy/blob/master/perlin2d.py
# and ported again from https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
# Bug: res divide d

import math
import random

import torch


def rand_perlin_2d(shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1]), indexing="ij"), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def tile_grads(slice1, slice2):
        return gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)

    def dot(grad, shift):
        return (
            torch.stack((grid[: shape[0], : shape[1], 0] + shift[0], grid[: shape[0], : shape[1], 1] + shift[1]), dim=-1)
            * grad[: shape[0], : shape[1]]
        ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def rand_perlin_2d_mask(shape, hills=4, percent: float | tuple[float, float] = 0.5):
    noise = rand_perlin_2d(shape, (hills, hills))
    noise -= torch.min(noise)
    noise /= torch.max(noise)
    if isinstance(percent, tuple):
        percent = torch.rand(1).item() * (percent[1] - percent[0]) + percent[0]
    noise[noise < percent] = 0
    noise[noise != 0] = 1
    return noise


def rand_perlin_3d(shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])

    grid = torch.stack(torch.meshgrid(
        torch.arange(0, res[0], delta[0]),
        torch.arange(0, res[1], delta[1]),
        torch.arange(0, res[2], delta[2]),
        indexing="ij"), dim=-1) % 1
    
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles), torch.sin(angles)), dim=-1)

    def tile_grads(slice1, slice2, slice3):
        return gradients[slice1[0]:slice1[1], slice2[0]:slice2[1], slice3[0]:slice3[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1).repeat_interleave(d[2], 2)

    def dot(grad, shift):
        return (
            torch.stack((
                grid[:shape[0], :shape[1], :shape[2], 0] + shift[0],
                grid[:shape[0], :shape[1], :shape[2], 1] + shift[1],
                grid[:shape[0], :shape[1], :shape[2], 2] + shift[2]
            ), dim=-1) * grad[:shape[0], :shape[1], :shape[2]]
        ).sum(dim=-1)

    n000 = dot(tile_grads([0, -1], [0, -1], [0, -1]), [0, 0, 0])
    n100 = dot(tile_grads([1, None], [0, -1], [0, -1]), [-1, 0, 0])
    n010 = dot(tile_grads([0, -1], [1, None], [0, -1]), [0, -1, 0])
    n110 = dot(tile_grads([1, None], [1, None], [0, -1]), [-1, -1, 0])
    n001 = dot(tile_grads([0, -1], [0, -1], [1, None]), [0, 0, -1])
    n101 = dot(tile_grads([1, None], [0, -1], [1, None]), [-1, 0, -1])
    n011 = dot(tile_grads([0, -1], [1, None], [1, None]), [0, -1, -1])
    n111 = dot(tile_grads([1, None], [1, None], [1, None]), [-1, -1, -1])
    
    t = fade(grid[:shape[0], :shape[1], :shape[2]])
    
    n0 = torch.lerp(torch.lerp(n000, n100, t[..., 0]), torch.lerp(n010, n110, t[..., 0]), t[..., 1])
    n1 = torch.lerp(torch.lerp(n001, n101, t[..., 0]), torch.lerp(n011, n111, t[..., 0]), t[..., 1])
    return math.sqrt(3) * torch.lerp(n0, n1, t[..., 2])


