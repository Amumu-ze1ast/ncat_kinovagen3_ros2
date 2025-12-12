#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp> // For converting tf2::Quaternion to geometry_msgs::msg::Quaternion

// Alias for simpler code
namespace mgi = moveit::planning_interface;

// Define the planning group (You confirmed this is "manipulator")
static const std::string PLANNING_GROUP = "manipulator"; 
static const rclcpp::Logger LOGGER = rclcpp::get_logger("SequentialPlannerCpp");

int main(int argc, char** argv)
{
  // 1. Initialize ROS 2
  rclcpp::init(argc, argv);
  // Create a Node with options for MoveIt
  auto const node = std::make_shared<rclcpp::Node>(
      "sequential_planner_cpp",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  // 2. Start spinning in a background thread for MoveIt to work
  // This is required so the MoveGroupInterface can communicate with the MoveIt planning services
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread([&executor]() { executor.spin(); }).detach();
  
  // 3. Define the MoveGroupInterface
  try {
      mgi::MoveGroupInterface move_group(node, PLANNING_GROUP);
      RCLCPP_INFO(LOGGER, "MoveGroupInterface initialized for group: %s", PLANNING_GROUP.c_str());
      
      // Configure Planning Settings
      move_group.setPlanningTime(5.0);
      move_group.setNumPlanningAttempts(5);

      // 4. Define Waypoints and Fixed Orientation
      // Waypoints (x, y, z) - Meters
      std::vector<std::array<double, 3>> given_points = {
          {0.3, 0.3, 0.3},   // Point A
          {0.35, 0.35, 0.35}, // Point B
          {0.4, 0.2, 0.3}    // Point C
      };

      // Fixed orientation (Roll=0, Pitch=180 deg, Yaw=0, e.g., pointing straight down)
      tf2::Quaternion orientation_tf;
      orientation_tf.setRPY(0, M_PI, 0); 
      
      geometry_msgs::msg::Quaternion fixed_orientation;
      fixed_orientation = tf2::toMsg(orientation_tf);
      

      // 5. Sequential Motion Loop
      for (size_t i = 0; i < given_points.size(); ++i) {
        auto point = given_points[i];
        RCLCPP_INFO(LOGGER, "--- Commanding Point %zu: (%.3f, %.3f, %.3f) ---", i + 1, point[0], point[1], point[2]);

        // Define Target Pose
        geometry_msgs::msg::Pose target_pose;
        target_pose.position.x = point[0];
        target_pose.position.y = point[1];
        target_pose.position.z = point[2];
        target_pose.orientation = fixed_orientation;

        // Plan and Execute (MoveIt handles the continuous seeding using the current robot state)
        move_group.setPoseTarget(target_pose);
        
        mgi::MoveGroupInterface::Plan my_plan;
        
        // Planning (Finds IK solution and path, using BioIK if configured)
        bool success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);

        if (success) {
          RCLCPP_INFO(LOGGER, "Plan successful. Executing motion...");
          // Execution
          move_group.execute(my_plan);
          // Important: Clear the target for the next loop iteration
          move_group.clearPoseTarget(); 
        } else {
          RCLCPP_ERROR(LOGGER, "Failed to find plan for point %zu.", i + 1);
          break;
        }
        
        // Optional pause between moves
        rclcpp::sleep_for(std::chrono::milliseconds(500)); 
      }

      RCLCPP_INFO(LOGGER, "Sequential motion complete.");
  } catch (const std::exception& e) {
      RCLCPP_ERROR(LOGGER, "Exception during MoveGroup setup or execution: %s", e.what());
  }


  rclcpp::shutdown();
  return 0;
}