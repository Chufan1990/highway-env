import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.highway_env import HighwayEnvFast


class CustomHighwayEnvFast(HighwayEnvFast):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
            "longitudinal_position_offset": 100,
            "feature_collision_check_time": 1.0
        })
        return cfg
        
    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # for vehicle in self.road.vehicles:
        #     if vehicle not in self.controlled_vehicles:
        #         vehicle.position[0] -= self.config["longitudinal_position_offset"]

        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if vehicle not in self.controlled_vehicles]

        for other in self.road.vehicles:
            other.position[0] -= self.config["longitudinal_position_offset"]
 
        for agent in self.controlled_vehicles:
            self.road.vehicles = [vehicle for vehicle in self.road.vehicles if not self._is_colliding(vehicle, agent, self.config["feature_collision_check_time"])]

        for agent in self.controlled_vehicles:
            self.road.vehicles.append(agent)

                
    @classmethod
    def _is_colliding(cls, this, other, dt):
        # Fast spherical pre-check
        if np.linalg.norm(other.position - this.position) > this.diagonal + this.speed * dt:
            return False
        # Accurate rectangular check
        intersecting, will_intersect, _ = utils.are_polygons_intersecting(this.polygon(), other.polygon(), this.velocity * dt, other.velocity * dt)
        return intersecting or will_intersect


register(
    id='highway-fast-v1',
    entry_point='highway_env.envs:CustomHighwayEnvFast',
)