from typing import Tuple

import chex
import esquilax
import jax
import jax.numpy as jnp

from . import data_types


@esquilax.transforms.spatial(
    10,
    (jnp.add, jnp.add, jnp.add, jnp.add),
    (0, jnp.zeros(2), 0.0, 0.0),
    include_self=False,
)
def observe(
    _k: chex.PRNGKey,
    _params: data_types.EnvParams,
    a: data_types.Boid,
    b: data_types.Boid,
):
    dh = esquilax.utils.shortest_vector(a.heading, b.heading, length=2 * jnp.pi)
    return 1, b.position, b.speed, dh


@esquilax.transforms.amap
def flatten_observations(_k: chex.PRNGKey, params: data_types.EnvParams, observations):
    boid, n_nb, x_nb, s_nb, h_nb = observations

    def obs_to_nbs():
        _x_nb = x_nb / n_nb
        _s_nb = s_nb / n_nb
        _h_nb = h_nb / n_nb

        dx = esquilax.utils.shortest_vector(boid.position, _x_nb)

        d = jnp.sqrt(jnp.sum(dx * dx)) / 0.1

        phi = jnp.arctan2(dx[1], dx[0]) + jnp.pi
        d_phi = (
            esquilax.utils.shortest_vector(boid.heading, phi, length=2 * jnp.pi)
            / jnp.pi
        )

        dh = _h_nb / jnp.pi
        ds = (_s_nb - boid.speed) / (params.max_speed - params.min_speed)

        return jnp.array([d, d_phi, dh, ds])

    return jax.lax.cond(
        n_nb > 0,
        obs_to_nbs,
        lambda: jnp.array([-1.0, 0.0, 0.0, -1.0]),
    )


@esquilax.transforms.amap
def update_velocity(
    _k: chex.PRNGKey,
    params: data_types.EnvParams,
    x: Tuple[chex.Array, data_types.Boid],
):
    actions, boid = x
    rotation = actions[0] * params.max_rotate * jnp.pi
    acceleration = actions[1] * params.max_accelerate

    new_heading = (boid.heading + rotation) % (2 * jnp.pi)
    new_speeds = jnp.clip(
        boid.speed + acceleration,
        min=params.min_speed,
        max=params.max_speed,
    )

    return new_heading, new_speeds


@esquilax.transforms.amap
def move(_key: chex.PRNGKey, _params: data_types.EnvParams, x):
    pos, heading, speed = x
    d_pos = jnp.array([speed * jnp.cos(heading), speed * jnp.sin(heading)])
    return (pos + d_pos) % 1.0


def vision_model(n: int, n_bins=10):
    @esquilax.transforms.spatial(
        n_bins,
        jnp.minimum,
        jnp.ones((n,)),
        include_self=False,
    )
    def view(
        _k: chex.PRNGKey,
        params: data_types.EnvParams,
        a: data_types.Boid,
        b: data_types.Boid,
    ):
        rays = jnp.linspace(
            -params.view_angle * jnp.pi,
            params.view_angle * jnp.pi,
            n,
            endpoint=True,
        )
        dx = esquilax.utils.shortest_vector(a.position, b.position)
        d = jnp.sqrt(jnp.sum(dx * dx))
        phi = jnp.arctan2(dx[1], dx[0]) % (2 * jnp.pi)
        dh = esquilax.utils.shortest_vector(phi, a.heading, 2 * jnp.pi)

        angular_width = jnp.arctan2(params.agent_radius, d)
        left = dh - angular_width
        right = dh + angular_width

        obs = jnp.where(jnp.logical_and(left < rays, rays < right), n_bins * d, 1.0)
        return obs

    return view
