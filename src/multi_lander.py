import math
from typing import TYPE_CHECKING, Optional
import numpy as np

import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` '
        + 'followed by `pip install "gymnasium[box2d]"`'
    ) from e

if TYPE_CHECKING:
    import pygame

FPS = 50
SCALE = 30.0  # affects how fast the game is, forces should be adjusted as well
MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17,0), (-17, -10), (+17, -10), (+17,0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12
MAIN_ENGINE_Y_LOCATION = (
    4  # The Y location of the main engine on the body of the Lander.
)

VIEWPORT_W = 600
VIEWPORT_H = 400


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        for lander in self.env.landers:
            if (
                lander == contact.fixtureA.body
                or lander == contact.fixtureB.body
            ):
                self.env.game_over = True
        for leg in self.env.legs:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True

    def EndContact(self, contact):
        for leg in self.env.legs:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False


class SequentialEnv(AECEnv, EzPickle):
    r"""
    This is a rewrite of the Farama foundation Lunar Lander environment from:
        https://gymnasium.farama.org/environments/box2d/lunar_lander/
    Adapted for a AEC type petting zoo environment.
    The action space and physics remain the same.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "multi_lander_v0",
        "is_parallelizable": False,
        "render_fps": FPS,
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
            continuous: bool = False,
            gravity: float = -10.0,
            enable_wind: bool = False,
            wind_power: float = 15.0,
            turbulence_power: float = 1.5,
            num_landers: int = 2,
            *args, **kwargs
    ):
        EzPickle.__init__(
            self,
            render_mode,
            continuous,
            gravity,
            enable_wind,
            wind_power,
            turbulence_power, 
            *args, **kwargs
        )
        AECEnv.__init__(self)
        self.render_mode = render_mode
        self.np_random = np.random

        # Value Checkers
        assert (-12.0 < gravity and gravity < 0.0
            ), f"gravity (current value: {gravity}) must be between -12 and 0"
        self.gravity = gravity

        if 0.0 > wind_power or wind_power > 20.0:
            logger.warn(
                "wind_power value is recommended to be between 0.0 and 20.0, " +
                f"(current value: {wind_power})"
            )
        self.wind_power = wind_power

        if 0.0 > turbulence_power or turbulence_power > 2.0:
            logger.warn(
                "turbulence_power value is recommended to be between 0.0 and " +
                f"2.0, (current value: {turbulence_power})"
            )

        # These are bounds for position realistically the environment
        # should have ended long before we reach more than 50% outside
        low = np.array(
            [
                -2.5,  # x coordinate
                -2.5,  # y coordinate
                -10.0, # velocity bounds is 5x rated speed
                -10.0,
                -2 * math.pi,
                -10.0,
                -0.0,
                -0.0,
            ]).astype(np.float32)
        high = np.array(
            [
                2.5,  # x coordinate
                2.5,  # y coordinate
                10.0, # velocity bounds is 5x rated speed
                10.0,
                2 * math.pi,
                10.0,
                1.0,
                1.0,
            ]).astype(np.float32)

        # Environmental Variables
        self.turbulence_power = turbulence_power
        self.enable_wind = enable_wind
        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True
        self.world = Box2D.b2World(gravity=(0, gravity))
        self.moon = None
        self.particles = []
        self.landers = []
        self.legs = []
        self.prev_reward = None
        self.continuous = continuous

        # Agents
        self.possible_agents = ["lander_" + str(i) for i in range(num_landers)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(zip(self.agents, 
                                           list(range(self.num_agents))))
        self._agent_selector = agent_selector(self.agents)

        # Spaces
        self.observation_space = spaces.Box(low, high)
        self.observation_spaces = dict(zip(self.agents, 
                                    [self.observation_space]*self.num_agents))
        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. 
            #               Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right 
            #               engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)
        self.action_spaces = dict(zip(self.agents, 
                                      [self.action_space]*self.num_agents))

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def convert_to_dict(self, list_of_list):
        return dict(zip(self.agents, list_of_list))

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            pygame.QUIT
            self.isopen = False

    def create_landers(self):
        landers = []
        for n in range(self.num_agents):
            lander: Box2D.b2Body = self.world.CreateDynamicBody(
                position=((n+1)/(self.num_agents+1) * VIEWPORT_W / SCALE, 
                          VIEWPORT_H / SCALE),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[(x/SCALE, y/SCALE) for x, y in LANDER_POLY]),
                    density=5.0,
                    friction=0.1,
                    categoryBits=0x0010,
                    #maskBits=0x001,  # uncomment for : collide only with ground
                    restitution=0.0,
                ),  # 0.99 bouncy
            )
            lander.color1 = ((128), (102+75*(n))%255, (230))
            lander.color2 = ((77), (77), (128))
            landers.append(lander)
        self.landers = landers

        legs = []
        for n,lander in enumerate(self.landers):
            for i in [-1, +1]:
                leg = self.world.CreateDynamicBody(
                    position=(lander.position[0] - i * LEG_AWAY / SCALE, 
                              lander.position[1]),
                    angle=(i * 0.05),
                    fixtures=fixtureDef(
                        shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                        density=1.0,
                        restitution=0.0,
                        categoryBits=0x0020,
                        #maskBits=0x001,
                    ),
                )
                leg.ground_contact = False
                leg.color1 = ((128), (102+75*(n))%255, (230))
                leg.color2 = ((77), (77), (128))
                rjd = revoluteJointDef(
                    bodyA=lander,
                    bodyB=leg,
                    localAnchorA=(0, 0),
                    localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=LEG_SPRING_TORQUE,
                    motorSpeed=+0.3 * i,  # low enough not to jump into the sky
                )
                if i == -1:
                    rjd.lowerAngle = (
                        +0.9 - 0.5
                    )  # Valid angle of travel for the legs
                    rjd.upperAngle = +0.9
                else:
                    rjd.lowerAngle = -0.9
                    rjd.upperAngle = -0.9 + 0.5
                leg.joint = self.world.CreateJoint(rjd)
                legs.append(leg)
        self.legs = legs

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3,
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all_particle):
        while self.particles and (all_particle or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))


    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying render mode."
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")')
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[box2d]"`'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        pygame.transform.scale(self.surf, (SCALE, SCALE))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
            obj.color2 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )

        self._clean_particles(False)

        for p in self.sky_polys:
            scaled_poly = []
            for coord in p:
                scaled_poly.append((coord[0] * SCALE, coord[1] * SCALE))
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, (0, 0, 0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf,color=obj.color1,points=path)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    pygame.draw.aalines(
                        self.surf, color=obj.color2, points=path, closed=True)

                for x in [self.helipad_x1, self.helipad_x2]:
                    x = x * SCALE
                    flagy1 = self.helipad_y * SCALE
                    flagy2 = flagy1 + 50
                    pygame.draw.line(
                        self.surf,
                        color=(255, 255, 255),
                        start_pos=(x, flagy1),
                        end_pos=(x, flagy2),
                        width=1,
                    )
                    pygame.draw.polygon(
                        self.surf,
                        color=(204, 204, 0),
                        points=[
                            (x, flagy2),
                            (x, flagy2 - 10),
                            (x + 25, flagy2 - 5),
                        ],
                    )
                    gfxdraw.aapolygon(
                        self.surf,
                        [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
                        (204, 204, 0),
                    )

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )
    # End Render()

    def _destroy(self):
        if not self.moon:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        for lander in self.landers:
            self.world.DestroyBody(lander)
            lander = None
        for i in range(len(self.legs)):
            self.world.DestroyBody(self.legs[i])


    def reset(self,*,seed: Optional[int] = None,options: Optional[dict] = None):
        self._destroy()
        # Issue: https://github.com/Farama-Foundation/Gymnasium/issues/728 
        # self._destroy() is not enough to clean(reset), workaround is 
        # to create a totally new world for self.reset()
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = [None]*self.num_agents

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Create Terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1,p2],density=0,friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        if self.enable_wind:  # Initialize wind pattern based on index
            self.wind_idx = self.np_random.integers(-9999, 9999)
            self.torque_idx = self.np_random.integers(-9999, 9999)

        self.create_landers()

        for lander in self.landers:
            lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )
        self.drawlist = self.legs + self.landers

        if self.render_mode == "human": self.render()
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        return self.observe()
    # End reset()


    def observe(self, agent=None):
        if not agent : agent = self.agent_selection
        id = self.agent_name_mapping[agent]
        lander = self.landers[id]
        pos = lander.position
        vel = lander.linearVelocity

        observation = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - (self.helipad_y + LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
            vel.x * (VIEWPORT_W/SCALE/2) / FPS,
            vel.y * (VIEWPORT_H/SCALE/2) / FPS,
            lander.angle,
            20.0 * lander.angularVelocity / FPS,
            1.0 if self.legs[id*2+0].ground_contact else 0.0,
            1.0 if self.legs[id*2+1].ground_contact else 0.0,
        ]
        assert len(observation) == 8
        return observation

    def latest_reward_state(self, agent, state):
        reward = 0
        id = self.agent_name_mapping[agent]
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6] + 10 * state[7]
        )   # And ten points for legs contact, the idea is if you
            # lose contact again after landing, you get negative reward
        if self.prev_shaping[id] is not None:
            reward = shaping - self.prev_shaping[id]
        self.prev_shaping[id] = shaping
        return reward


    def step(self, action):
        assert self.landers is not None, "You forgot to call reset()"
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # If one agent has terminated this accepts a None action,
            # which otherwise errors, handles stepping to the next agent
            obs = self.observe(self.agent_selection)
            self.agent_selection = self._agent_selector.next()
            return obs, 0, True, False, {}

        # Update wind and apply to the lander
        if self.enable_wind and not any([l.ground_contact for l in self.legs]):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi * 0.01 * self.wind_idx))
                ) * self.wind_power
            )
            self.wind_idx += 1

            for lander in self.landers:
                lander.ApplyForceToCenter(
                    (wind_mag, 0.0),
                    True,
                )

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = (
                math.tanh(
                    math.sin(0.02 * self.torque_idx)
                    + (math.sin(math.pi * 0.01 * self.torque_idx))
                ) * self.turbulence_power
            )
            self.torque_idx += 1
            
            for lander in self.landers:
                lander.ApplyTorque(
                    torque_mag,
                    True,
                )

        # For current agent:
        agent = self.agent_selection
        lander = self.landers[self.agent_name_mapping[agent]]
        # Check action validity
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float64)
        else:
            assert self.action_spaces[agent].contains(
            #assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "

        # Tip is the (X and Y) components of the rotation of the lander.
        tip = (math.sin(lander.angle), math.cos(lander.angle))

        # Side is the (-Y and X) components of the rotation of the lander.
        side = (-tip[1], tip[0])

        # Generate two random numbers between -1/SCALE and 1/SCALE.
        dispersion = [self.np_random.uniform(-1.0,+1.0)/SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0

            # 4 is move a bit downwards, +-2 for randomness
            # The components of the impulse to be applied by the main engine.
            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )
            impulse_pos = (lander.position[0] + ox, lander.position[1] + oy)
            if self.render_mode is not None:
                # particles are just a decoration, with no physics impact, 
                # so don't add them when not rendering
                p = self._create_particle(
                    3.5,  # 3.5 is here to make particle speed adequate
                    impulse_pos[0],
                    impulse_pos[1],
                    m_power,
                )
                p.ApplyLinearImpulse(
                    (
                        ox * MAIN_ENGINE_POWER * m_power,
                        oy * MAIN_ENGINE_POWER * m_power,
                    ),
                    impulse_pos,
                    True,
                )
            lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, 
                 -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            # Orientation/Side engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                # action = 1 is left, action = 3 is right
                direction = action - 2
                s_power = 1.0

            # The components of the impulse to be applied by the side engines.
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )

            # The constant 17 is presumably meant to be SIDE_ENGINE_HEIGHT.
            # However, SIDE_ENGINE_HEIGHT is defined as 14, causing the 
            # position of the thrust on the body of the lander to change, 
            # depending on the orientation of the lander. This results in 
            # an orientation dependent torque being applied to the lander.

            impulse_pos = (
                lander.position[0] + ox - tip[0] * 17 / SCALE,
                lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            if self.render_mode is not None:
                # particles are just decoration, with no impact on the physics,
                # so don't add them when not rendering
                p = self._create_particle(0.7, impulse_pos[0], 
                                          impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (ox * SIDE_ENGINE_POWER * s_power,
                     oy * SIDE_ENGINE_POWER * s_power,),
                    impulse_pos,
                    True,
                )
            lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, 
                 -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        # Update Positions
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        observation = self.observe(agent)
        reward = self.latest_reward_state(agent, observation)
        reward -= m_power * 0.30 # Deduct cost of fuel
        reward -= s_power * 0.03 # Deduct cost of fuel

        if self.game_over or abs(observation[0]) >= 1.0:
            self.terminations[self.agent_selection] = True
            reward = -100
        if not lander.awake:
            self.terminations[self.agent_selection] = True
            reward = +100
        terminated = self.terminations[self.agent_selection]

        self.rewards[agent] = reward
        self._accumulate_rewards()

        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` 
        return observation, reward, terminated, False, {}


class Parallel_Env(SequentialEnv):
    def observe(self):
        obs_list = {}
        for agent in self.agents:
            obs_list[agent] = super().observe(agent)
        return obs_list

    def last(self):
        return self.observe()

    def step(self, action_list):
        self._clear_rewards()
        assert self.landers is not None, "You forgot to call reset()"

        # Update wind and apply to the lander
        if self.enable_wind and not any([l.ground_contact for l in self.legs]):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi * 0.01 * self.wind_idx))
                ) * self.wind_power
            )
            self.wind_idx += 1

            for lander in self.landers:
                lander.ApplyForceToCenter(
                    (wind_mag, 0.0),
                    True,
                )
            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = (
                math.tanh(
                    math.sin(0.02 * self.torque_idx)
                    + (math.sin(math.pi * 0.01 * self.torque_idx))
                ) * self.turbulence_power
            )
            self.torque_idx += 1
            
            for lander in self.landers:
                lander.ApplyTorque(
                    torque_mag,
                    True,
                )

        for agent in self.agents:
            # For current agent:
            lander = self.landers[self.agent_name_mapping[agent]]
            action = action_list[agent]
            # Check action validity
            if self.continuous:
                action = np.clip(action, -1, +1).astype(np.float64)
            else:
                assert self.action_spaces[agent].contains(
                #assert self.action_space.contains(
                    action
                ), f"{action!r} ({type(action)}) invalid "

            # Tip is the (X and Y) components of the rotation of the lander.
            tip = (math.sin(lander.angle), math.cos(lander.angle))

            # Side is the (-Y and X) components of the rotation of the lander.
            side = (-tip[1], tip[0])

            # Generate two random numbers between -1/SCALE and 1/SCALE.
            dispersion = [self.np_random.uniform(-1.0,+1.0)/SCALE for _ in range(2)]

            m_power = 0.0
            if (self.continuous and action[0] > 0.0) or (
                not self.continuous and action == 2
            ):
                # Main engine
                if self.continuous:
                    m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                    assert m_power >= 0.5 and m_power <= 1.0
                else:
                    m_power = 1.0

                # 4 is move a bit downwards, +-2 for randomness
                # The components of the impulse to be applied by the main engine.
                ox = (
                    tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                    + side[0] * dispersion[1]
                )
                oy = (
                    -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                    - side[1] * dispersion[1]
                )
                impulse_pos = (lander.position[0] + ox, lander.position[1] + oy)
                if self.render_mode is not None:
                    # particles are just a decoration, with no physics impact, 
                    # so don't add them when not rendering
                    p = self._create_particle(
                        3.5,  # 3.5 is here to make particle speed adequate
                        impulse_pos[0],
                        impulse_pos[1],
                        m_power,
                    )
                    p.ApplyLinearImpulse(
                        (
                            ox * MAIN_ENGINE_POWER * m_power,
                            oy * MAIN_ENGINE_POWER * m_power,
                        ),
                        impulse_pos,
                        True,
                    )
                lander.ApplyLinearImpulse(
                    (-ox * MAIN_ENGINE_POWER * m_power, 
                    -oy * MAIN_ENGINE_POWER * m_power),
                    impulse_pos,
                    True,
                )

            s_power = 0.0
            if (self.continuous and np.abs(action[1]) > 0.5) or (
                not self.continuous and action in [1, 3]
            ):
                # Orientation/Side engines
                if self.continuous:
                    direction = np.sign(action[1])
                    s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                    assert s_power >= 0.5 and s_power <= 1.0
                else:
                    # action = 1 is left, action = 3 is right
                    direction = action - 2
                    s_power = 1.0

                # The components of the impulse to be applied by the side engines.
                ox = tip[0] * dispersion[0] + side[0] * (
                    3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
                )
                oy = -tip[1] * dispersion[0] - side[1] * (
                    3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
                )

                # The constant 17 is presumably meant to be SIDE_ENGINE_HEIGHT.
                # However, SIDE_ENGINE_HEIGHT is defined as 14, causing the 
                # position of the thrust on the body of the lander to change, 
                # depending on the orientation of the lander. This results in 
                # an orientation dependent torque being applied to the lander.

                impulse_pos = (
                    lander.position[0] + ox - tip[0] * 17 / SCALE,
                    lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
                )
                if self.render_mode is not None:
                    # particles are just decoration, with no impact on the physics,
                    # so don't add them when not rendering
                    p = self._create_particle(0.7, impulse_pos[0], 
                                            impulse_pos[1], s_power)
                    p.ApplyLinearImpulse(
                        (ox * SIDE_ENGINE_POWER * s_power,
                        oy * SIDE_ENGINE_POWER * s_power,),
                        impulse_pos,
                        True,
                    )
                lander.ApplyLinearImpulse(
                    (-ox * SIDE_ENGINE_POWER * s_power, 
                    -oy * SIDE_ENGINE_POWER * s_power),
                    impulse_pos,
                    True,
                )
                self.rewards[agent] -= m_power * 0.30 # Deduct cost of fuel
                self.rewards[agent] -= s_power * 0.03 # Deduct cost of fuel
        """ End Agent Loop """

        # Update Positions
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        #observation = self.observe(agent)
        obs_list = self.observe()

        for agent in self.agents:
            obs = obs_list[agent]
            self.rewards[agent] += self.latest_reward_state(agent, obs)
            if self.game_over or abs(obs[0]) >= 1.0:
                self.terminations[agent] = True
                self.rewards[agent] -= 100
            if not lander.awake:
                self.terminations[agent] = True
                self.rewards[agent] += 100

        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` 
        return obs_list, self.rewards, all(self.terminations), False, {}


def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
            s[0] is the horizontal coordinate
            s[1] is the vertical coordinate
            s[2] is the horizontal speed
            s[3] is the vertical speed
            s[4] is the angle
            s[5] is the angular speed
            s[6] 1 if first leg has contact, else 0
            s[7] 1 if second leg has contact, else 0

    Returns:
         a: The heuristic to be fed into the step function defined above to 
            determine the next step and reward.
    """

    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        s[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(s[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    if env.unwrapped.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a

def demo_heuristic_lander(env, reps=1, seed=None, render=False):
    total_reward = 0
    steps = 0
    for _ in range(reps):
        s = env.reset(seed=seed)
        while True:
            s = env.last()[0]
            a = heuristic(env, s)
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r

            if render:
                still_open = env.render()
                if still_open is False:
                    break

            if steps % 20 == 0 or terminated or truncated:
                print("observations:", " ".join([f"{x:+0.2f}" for x in s]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
                print(env.terminations.values())
            steps += 1
            # if terminated or truncated:
            if all(env.terminations.values()):
                break
    if render:
        env.close()
    return total_reward

def demo_parallel_heuristic(env, reps=1, seed=None, render=False):
    total_reward = 0
    steps = 0
    for _ in range(reps):
        _ = env.reset(seed=seed)
        obs = env.last()
        while True:
            actions = {agent: heuristic(env, obs[agent]) for agent in env.agents}
            obs, r, terminated, truncated, info = env.step(actions)
            total_reward += sum(r.values())

            if render:
                still_open = env.render()
                if still_open is False:
                    break

            if steps % 20 == 0 or terminated or truncated:
                for agent in env.agents:
                    print(f"{agent} observations:", " ".join([f"{x:+0.2f}" for x in obs[agent]]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
                print(env.terminations.values())
            steps += 1
            # if terminated or truncated:
            if all(env.terminations.values()):
                break
    if render:
        env.close()
    return total_reward

def env(**kwargs):
    return SequentialEnv(**kwargs)

def parallel_env(**kwargs):
    return Parallel_Env(**kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=
                                     'A multi-agent version of Lunar Lander')
    parser.add_argument('-s', '--sequential', action="store_true", 
                        help='Iterate as AEC environment')
    parser.add_argument('-n', '--num_landers', type=int, default=2,
                        help='number of landers')
    parser.add_argument('-d', '--demo_iters', type=int, default=1,
                        help='number of landers')
    args = parser.parse_args()

    if args.sequential:
        _env = env(render_mode="human", num_landers=args.num_landers)
        demo_heuristic_lander(_env, reps=args.demo_iters, render=True)
    else:
        _env = parallel_env(render_mode="human", num_landers=args.num_landers)
        demo_parallel_heuristic(_env, reps=args.demo_iters, render=True)
