#!/usr/bin/env python3

import numpy as np 
from cw3q2.iiwa14DynStudent import Iiwa14DynamicRef as Student
from cw3q2.iiwa14DynKDL import Iiwa14DynamicKDL as KDL

def compare_matrices(mat1, mat2, tol=1e-6):
    return np.allclose(mat1, mat2, atol=tol)

def main():
    try: 
        print("Starting comparison...")
        student = Student()
        kdl = KDL()
        print("Classes initialised successfully.")

        joint_readings = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        joint_velocities = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        print("Joint readings:", joint_readings)
        print("Joint velocities:", joint_velocities)

        print("Computing B matrices...")
        B_student = student.get_B(joint_readings)
        B_kdl = kdl.get_B(joint_readings)
        print("B matrices are equal:", compare_matrices(B_student, B_kdl))

        print("Cmputing C matrices...")
        C_student = student.get_C_times_qdot(joint_readings, joint_velocities)
        C_kdl = kdl.get_C_times_qdot(joint_readings, joint_velocities)
        print("C matrices are equal", compare_matrices(C_student, C_kdl))

        print("Computing G vectors...")
        G_student = student.get_G(joint_readings, joint_velocities)
        G_kdl = kdl.get_G(joint_readings, joint_velocities)
        print("G vectors are equal", compare_matrices(G_student, G_kdl))

    except Exception as e:
        print("An error occurred:", e)

    if __name__ == "__main__":
        main()
        