#!/usr/bin/env python3
import rospy
import numpy as np
import actionlib
from enum import Enum
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import tf.transformations as tf

# Dynamic reconfigure
from dynamic_reconfigure.server import Server
from girona500_inspection_trajectories.cfg import InspectionPlannerConfig

# Service (trigger-only)
from girona500_inspection_trajectories.srv import (
    PlanInspectionPath,
    PlanInspectionPathResponse
)


from girona500_inspection_trajectories.msg import (
    ExecutePlaneInspectionAction,
    ExecutePlaneInspectionFeedback,
    ExecutePlaneInspectionResult
)


# ============================================================
# State machine
# ============================================================

class InspectionState(Enum):
    IDLE = 0
    PREVIEW = 1
    PLANNED = 2
    EXECUTING = 3


# ============================================================
# Plane Inspection Node
# ============================================================

class PlaneInspectionNode:

    def __init__(self):
        rospy.loginfo("[PlaneInspection] Initializing")

        self.state = InspectionState.IDLE

        self.preview_path = None
        self.planned_path = None

        # Publishers
        self.preview_pub = rospy.Publisher(
            "plane_inspection/path_preview",
            Path,
            queue_size=1,
            latch=True
        )
        self.path_marker_pub = rospy.Publisher(
            "/plane_inspection/path_markers",
            MarkerArray,
            queue_size=1,
            latch=True
        )

        # Services
        self.plan_srv = rospy.Service(
            "plane_inspection/plan_path",
            PlanInspectionPath,
            self.plan_path_cb
        )

        # Dynamic reconfigure
        self.dyn_server = Server(
            InspectionPlannerConfig,
            self.dynamic_reconfigure_cb
        )

        rospy.loginfo("[PlaneInspection] Ready")

    # ========================================================
    # Utilities
    # ========================================================

    @staticmethod
    def normalize(v):
        n = np.linalg.norm(v)
        if n < 1e-8:
            raise ValueError("Zero-length vector")
        return v / n

    @staticmethod
    def project_onto_plane(v, n):
        return v - np.dot(v, n) * n
    
    # ========================================================
    # Publish Marker arrays (for RViz visualization)
    # ========================================================
    
    def publish_path_markers(self, path, active_index=0):
        ma = MarkerArray()

        for i, pose in enumerate(path.poses):
            m = Marker()

            m.header.frame_id = path.header.frame_id
            m.header.stamp = rospy.Time.now()

            m.ns = "plane_inspection_path"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            m.pose.position.x = pose.pose.position.x
            m.pose.position.y = pose.pose.position.y
            m.pose.position.z = pose.pose.position.z
            m.pose.orientation.w = 1.0

            # Tamaño
            m.scale.x = 0.18
            m.scale.y = 0.18
            m.scale.z = 0.18

            # 🎨 COLORES SEGÚN ESTADO
            if i == active_index:
                if self.state == InspectionState.PREVIEW:
                    # 🔵 Azul
                    m.color.r = 0.0
                    m.color.g = 0.0
                    m.color.b = 1.0
                elif self.state == InspectionState.PLANNED:
                    # 🟡 Amarillo
                    m.color.r = 1.0
                    m.color.g = 1.0
                    m.color.b = 0.0
                elif self.state == InspectionState.EXECUTING:
                    # 🔴 Rojo
                    m.color.r = 1.0
                    m.color.g = 0.0
                    m.color.b = 0.0
                else:
                    # fallback
                    m.color.r = 1.0
                    m.color.g = 1.0
                    m.color.b = 1.0

                m.color.a = 1.0

            else:
                # 🟢 Resto de waypoints
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 0.0
                m.color.a = 0.6

            ma.markers.append(m)

        self.path_marker_pub.publish(ma)

    # ========================================================
    # Dynamic reconfigure callback (PREVIEW MODE)
    # ========================================================

    def dynamic_reconfigure_cb(self, cfg, level):
        self.state = InspectionState.PREVIEW
        rospy.loginfo("[PlaneInspection] Dynamic reconfigure update")

        try:
            self.preview_path = self.generate_path_from_cfg(cfg)
            self.preview_pub.publish(self.preview_path)

            if self.preview_path.poses:
                self.publish_path_markers(self.preview_path, active_index=0)

            rospy.loginfo(
                "[PlaneInspection] Preview path published (%d poses)",
                len(self.preview_path.poses)
            )

        except Exception as e:
            rospy.logerr("[PlaneInspection] Preview failed: %s", str(e))

        return cfg

    # ========================================================
    # Trigger PLAN service (commit preview → planned)
    # ========================================================
    def plan_path_cb(self, req):
        self.state = InspectionState.PLANNED
        rospy.loginfo("[PlaneInspection] Plan trigger received")

        if self.preview_path is None:
            return PlanInspectionPathResponse(
                success=False,
                message="No preview path available"
            )

        self.planned_path = self.preview_path
        self.preview_pub.publish(self.planned_path)

        if self.planned_path.poses:
            self.publish_path_markers(self.planned_path, active_index=0)

        rospy.loginfo(
            "[PlaneInspection] Path committed (%d poses)",
            len(self.planned_path.poses)
        )

        return PlanInspectionPathResponse(
            success=True,
            message="Path planned successfully"
        )

    # ========================================================
    # Path generation
    # ========================================================

    def generate_path_from_cfg(self, cfg):
        """
        Generate a raster inspection path on a plane using
        dynamic reconfigure parameters.
        """

        # --- Plane definition ---
        n = self.normalize(np.array([cfg.normal_x,
                                     cfg.normal_y,
                                     cfg.normal_z]))

        u_raw = np.array([cfg.u_axis_x,
                          cfg.u_axis_y,
                          cfg.u_axis_z])

        u = self.normalize(self.project_onto_plane(u_raw, n))
        v = np.cross(n, u)

        origin = np.array([cfg.origin_x,
                           cfg.origin_y,
                           cfg.origin_z])

        # --- Discretization ---
        nu = cfg.number_steps
        nv = cfg.number_steps

        u_vals = np.linspace(0.0, cfg.u_amplitude, nu)
        v_vals = np.linspace(0.0, cfg.v_amplitude, nv)

        # --- Path message ---
        path = Path()
        path.header.frame_id = "world_ned"
        path.header.stamp = rospy.Time.now()

        zigzag = False
        for v_i in v_vals:
            row = u_vals if not zigzag else u_vals[::-1]
            zigzag = not zigzag

            for u_i in row:
                p = (origin
                     + u * u_i
                     + v * v_i
                     + n * cfg.normal_offset)

                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x = p[0]
                pose.pose.position.y = p[1]
                pose.pose.position.z = p[2]

                # Orientation: align yaw with u-axis projection
                yaw = np.arctan2(u[1], u[0])
                q = tf.quaternion_from_euler(0.0, 0.0, yaw)

                pose.pose.orientation.x = q[0]
                pose.pose.orientation.y = q[1]
                pose.pose.orientation.z = q[2]
                pose.pose.orientation.w = q[3]

                path.poses.append(pose)

        return path


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    rospy.init_node("plane_inspection_node")
    PlaneInspectionNode()
    rospy.spin()
