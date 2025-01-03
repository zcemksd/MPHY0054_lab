#!/usr/bin/env python3
print("Script started")
try:
    print("Importing numpy...")
    import numpy as np 
    print("Numpy imported successfully...")

    print("Importing Student class...")
    from cw3q2.iiwa14DynStudent import Iiwa14DynamicRef as Student
    print("Student class imported successfully...")

    print("Importing KDL class...")
    from cw3q2.iiwa14DynKDL import Iiwa14DynamicKDL as KDL
    print("KDL class imported successfully...")
    
except Exception as e:
    print("An error occurred during imports:", e)

def compare_matrices(mat1, mat2, tol=1e-4):
    print("Comparing matrices...")
    return np.allclose(mat1, mat2, atol=tol)

def main():
    print("Main function started")
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
        print("B matrix (Student):")
        print(B_student)
        print("B matrix (KDL):")
        print(B_kdl)
        print("B matrices are equal:", compare_matrices(B_student, B_kdl))

        print("Cmputing C matrices...")
        C_student = student.get_C_times_qdot(joint_readings, joint_velocities)
        C_kdl = kdl.get_C_times_qdot(joint_readings, joint_velocities)
        print("C matrix (Student):")
        print(C_student)
        print("C matrix (KDL):")
        print(C_kdl)
        print("C matrices are equal", compare_matrices(C_student, C_kdl))

        print("Computing G vectors...")
        G_student = student.get_G(joint_readings)
        G_kdl = kdl.get_G(joint_readings)
        print("G vector (Student):")
        print(G_student)
        print("G vector (KDL):")
        print(G_kdl)
        print("G vectors are equal", compare_matrices(G_student, G_kdl))

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
