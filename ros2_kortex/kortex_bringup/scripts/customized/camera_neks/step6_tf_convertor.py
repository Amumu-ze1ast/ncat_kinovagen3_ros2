#!/usr/bin/env python3
import numpy as np
import sys

def rpy_to_matrix(roll, pitch, yaw):
    """Convert roll, pitch, yaw (radians) to a 3x3 rotation matrix.
    Rotation order applied is Rx(roll) → Ry(pitch) → Rz(yaw)."""
    
    # Rotation Matrix for X (Roll)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
                   
    # Rotation Matrix for Y (Pitch)
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
                   
    # Rotation Matrix for Z (Yaw)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1]])
                   
    # Combined Rotation Matrix (R = Rz * Ry * Rx)
    return Rz @ Ry @ Rx # apply roll, then pitch, then yaw

if __name__ == "__main__":
    # --- Example TF from camera → base ---
    
    # Translation vector t (meters)
    t = np.array([-0.024, -0.408, 0.887])
    
    # Roll, pitch, yaw vector rpy (radians)
    rpy = np.array([-2.050, -0.001, -3.140])
    
    # Calculate the 3x3 Rotation Matrix R
    R = rpy_to_matrix(*rpy)

    # --- Example points in camera frame ---
    pt_cam = [
        np.array([-0.145, 0.137, 0.392]),
        # np.array([ 0.205, 0.019, 0.714]),
    ]

    # --- Transform each point: p_base = R * p_cam + t ---
    # R is the 3x3 rotation matrix
    # p_cam is the point in the camera frame
    # t is the translation vector
    pt_base = [R.dot(p) + t for p in pt_cam]
    # print("--- Coordinate Frame Transformation (Camera to Base) ---")
    for idx, (pc, pb) in enumerate(zip(pt_cam, pt_base), start=1):
        # Format the output for better readability
        pc_str = ", ".join(f"{val:.3f}" for val in pc)
        pb_str = ", ".join(f"{val:.3f}" for val in pb)
        
        print(f"Point {idx} in camera frame: [{pc_str}]")
        print(f"Point {idx} in base frame:   [{pb_str}]\n")

    print(f"Final Rotation Matrix R:\n{R}")