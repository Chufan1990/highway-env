from tabnanny import verbose
import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.highway_env import HighwayEnvFast
from highway_env.vehicle.objects import Landmark
from highway_env.envs.common.action import Action
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle import kinematics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.highway_env import HighwayEnvFast
from highway_env.vehicle.objects import Landmark

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
            "feature_collision_check_time": 1.0,
            "congest_reward": -0.8,
            "reward_congest_range": [0, 3],
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

    def vehicles_behind(self, vehicle: 'kinematics.Vehicle', distance: float = None, verbose: int = 0) \
            -> int:
        """
        Find the preceding and following vehicles of a given vehicle.

        :param vehicle: the vehicle whose neighbours must be found
        :param lane_index: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        v_rear = 0
        lane_index = vehicle.lane_index
        if not lane_index:
            return v_rear
        lane = self.road.network.get_lane(lane_index)
        s = self.road.network.get_lane(lane_index).local_coordinates(vehicle.position)[0]
        for v in self.road.vehicles + self.road.objects:
            if v is not vehicle and not isinstance(v, Landmark):  # self.network.is_connected_road(v.lane_index,
                # lane_index, same_lane=True):
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v, margin=1):
                    continue
                if s > s_v and abs(vehicle.lane_distance_to(v)) < distance if distance is not None else (self.config["reward_congest_range"][1] + 3)* Vehicle.LENGTH:
                    v_rear += 1
                    if verbose > 0:
                        print(f"blocked vehicles: {v}")
        return v_rear

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        self._vehicles_blocked = self.vehicles_behind(self.vehicle, 40.0)
        scaled_blocked_vehicles = utils.lmap(self._vehicles_blocked, self.config["reward_congest_range"], [0, 1])
        # reward = \
        #     + self.config["collision_reward"] * self.vehicle.crashed \
        #     + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
        #     + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
        #     + self.config["congest_reward"] * np.clip(scaled_blocked_vehicles, 0, 1)
        reward = self.config["collision_reward"] if self.vehicle.crashed else \
            self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
            + self.config["congest_reward"] * np.clip(scaled_blocked_vehicles, 0, 1) 
        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    @property
    def vehicles_blocked(self) -> int:
        return self._vehicles_blocked

register(
    id='highway-fast-v1',
    entry_point='highway_env.envs:CustomHighwayEnvFast',
)