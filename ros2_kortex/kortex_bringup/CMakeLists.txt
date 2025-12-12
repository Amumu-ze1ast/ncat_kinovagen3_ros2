cmake_minimum_required(VERSION 3.5)
project(kortex_bringup)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python executables
install(PROGRAMS
  scripts/test_huge_joint_space_jump.py
  scripts/customized/1joint_forward_once.py
  scripts/customized/2gripper_once.py
  scripts/customized/3joint_forward_multiple.py
  scripts/customized/4marker_multiple.py
  scripts/customized/camera_neks/step7_tf_all.py

  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
