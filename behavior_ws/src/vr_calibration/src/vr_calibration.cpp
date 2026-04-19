// vr_calibration.cpp  (Option B: Auto-capture + Update R_Adj, T_AD, T_BC, T_SA in one YAML write)
// v7: add saved_at timestamp comment + add t_sa_mode(keep/update) + add t_sa_max_delta_deg guard

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <array>
#include <string>
#include <mutex>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <ctime>   // for timestamp

// ================= Constants =================
static constexpr double kPi = 3.14159265358979323846;

static inline double deg2rad(double d){ return d * kPi / 180.0; }
static inline double rad2deg(double r){ return r * 180.0 / kPi; }

static inline double clampd(double x, double lo, double hi)
{
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

// local time string: "YYYY.MM.DD HH:MM"
static std::string nowLocalString()
{
  std::time_t t = std::time(nullptr);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  std::ostringstream oss;
  oss << std::setfill('0')
      << (tm.tm_year + 1900) << "."
      << std::setw(2) << (tm.tm_mon + 1) << "."
      << std::setw(2) << tm.tm_mday << " "
      << std::setw(2) << tm.tm_hour << ":"
      << std::setw(2) << tm.tm_min;
  return oss.str();
}

// ================= Rodrigues (rotvec -> R) =================
static void rotvecToRotMatRad(const std::array<double,3>& w,
                              std::array<double,9>& R)
{
  double th = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
  R = {1,0,0, 0,1,0, 0,0,1};
  if (th < 1e-12) return;

  double ux = w[0]/th, uy = w[1]/th, uz = w[2]/th;
  double s = std::sin(th), c = std::cos(th), v = 1.0 - c;

  R[0] = ux*ux*v + c;
  R[1] = ux*uy*v - uz*s;
  R[2] = ux*uz*v + uy*s;

  R[3] = uy*ux*v + uz*s;
  R[4] = uy*uy*v + c;
  R[5] = uy*uz*v - ux*s;

  R[6] = uz*ux*v - uy*s;
  R[7] = uz*uy*v + ux*s;
  R[8] = uz*uz*v + c;
}

static std::array<double,9> matMul3(const std::array<double,9>& A,
                                    const std::array<double,9>& B)
{
  std::array<double,9> C{};
  for (int r=0;r<3;r++){
    for (int c=0;c<3;c++){
      double s=0;
      for (int k=0;k<3;k++) s += A[r*3+k]*B[k*3+c];
      C[r*3+c]=s;
    }
  }
  return C;
}

static std::array<double,9> matT3(const std::array<double,9>& A)
{
  return {A[0],A[3],A[6],
          A[1],A[4],A[7],
          A[2],A[5],A[8]};
}

static double trace3(const std::array<double,9>& A)
{
  return A[0] + A[4] + A[8];
}

static double rotAngleBetweenRad(const std::array<double,9>& R_target,
                                 const std::array<double,9>& R_current)
{
  auto RtT  = matT3(R_target);
  auto Rrel = matMul3(RtT, R_current);
  double cosang = (trace3(Rrel) - 1.0) * 0.5;
  cosang = clampd(cosang, -1.0, 1.0);
  return std::acos(cosang);
}

// ================= Waypoint =================
struct Waypoint
{
  std::array<double,6> pose; // x y z wx wy wz
  double flag;               // last column
};

// ================= helper: Eigen rigid transform =================
static Eigen::Matrix4d makeT(const Eigen::Matrix3d& R, const Eigen::Vector3d& p)
{
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3,3>(0,0) = R;
  T.block<3,1>(0,3) = p;
  return T;
}

static Eigen::Matrix4d invT(const Eigen::Matrix4d& T)
{
  Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity();
  const Eigen::Matrix3d R = T.block<3,3>(0,0);
  const Eigen::Vector3d p = T.block<3,1>(0,3);
  Ti.block<3,3>(0,0) = R.transpose();
  Ti.block<3,1>(0,3) = -R.transpose()*p;
  return Ti;
}

static Eigen::Matrix<double,9,9> kron3(const Eigen::Matrix3d& A, const Eigen::Matrix3d& B)
{
  Eigen::Matrix<double,9,9> K;
  for (int i=0;i<3;i++){
    for (int j=0;j<3;j++){
      K.block<3,3>(3*i,3*j) = A(i,j) * B;
    }
  }
  return K;
}

static double rotDiffAngleDeg(const Eigen::Matrix3d& R1, const Eigen::Matrix3d& R2)
{
  Eigen::Matrix3d Rrel = R1.transpose() * R2;
  double cosang = (Rrel.trace() - 1.0) * 0.5;
  cosang = clampd(cosang, -1.0, 1.0);
  return rad2deg(std::acos(cosang));
}

// ================= Class =================
class VrCalibration : public rclcpp::Node
{
public:
  VrCalibration()
  : Node("vr_calibration_target_based"),
    steady_clock_(RCL_STEADY_TIME)
  {
    // ----------------------------
    // Files
    // ----------------------------
    waypoint_file_ =
      "/home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/vr_calibration/for_vr_calibration_point_v2.txt";

    ee_path_ =
      "/home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/vr_calibration/ur10_ee.txt";
    vr_path_ =
      "/home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/vr_calibration/ur10_vr.txt";

    // ----------------------------
    // Tunables
    // ----------------------------
    pos_enter_mm_ = 20.0;
    pos_exit_mm_  = 60.0;

    ori_enter_deg_ = 25.0;
    ori_exit_deg_  = 60.0;

    vel_thresh_mms_      = 15.0;  // mm/s
    angvel_thresh_dps_   = 8.0;   // deg/s

    hold_time_s_ = 0.35;

    cp_fresh_s_       = 1.0;
    vr_capture_age_s_ = 30.0;

    target_timeout_s_ = 300.0;
    loop_hz_          = 200.0;

    cp_unit_probe_N_ = 30;

    // ----------------------------
    // T_SA params
    // ----------------------------
    this->declare_parameter<double>("t_sa_w_des_z", 1.5707963267948966); // rad (≈ pi/2)
    this->declare_parameter<double>("t_sa_wait_timeout_s", 15.0);
    this->declare_parameter<double>("t_sa_hold_s", 0.25);
    this->declare_parameter<double>("t_sa_fresh_s", 1.0);

    // ✅ NEW: T_SA update mode
    // - keep   : YAML의 기존 T_SA 유지(덮어쓰기 방지, 기본)
    // - update : pre-phase에서 /calibrated_pose 기반으로 T_SA 재계산
    this->declare_parameter<std::string>("t_sa_mode", "keep"); // "keep" or "update"
    this->declare_parameter<double>("t_sa_max_delta_deg", 20.0); // update 시 old->new 변화량 제한

    t_sa_w_des_z_        = this->get_parameter("t_sa_w_des_z").as_double();
    t_sa_wait_timeout_s_ = this->get_parameter("t_sa_wait_timeout_s").as_double();
    t_sa_hold_s_         = this->get_parameter("t_sa_hold_s").as_double();
    t_sa_fresh_s_        = this->get_parameter("t_sa_fresh_s").as_double();

    t_sa_mode_           = this->get_parameter("t_sa_mode").as_string();
    std::transform(t_sa_mode_.begin(), t_sa_mode_.end(), t_sa_mode_.begin(), ::tolower);
    t_sa_max_delta_deg_  = this->get_parameter("t_sa_max_delta_deg").as_double();

    // ----------------------------
    // Waypoints
    // ----------------------------
    loadWaypointsAndDecideWpUnits();
    buildTargetIndices();

    // ---- output file refresh ----
    ee_ofs_.open(ee_path_, std::ios::out | std::ios::trunc);
    vr_ofs_.open(vr_path_, std::ios::out | std::ios::trunc);

    if (!ee_ofs_.is_open() || !vr_ofs_.is_open())
      throw std::runtime_error("Failed to open output files (truncate mode)");

    // ----------------------------
    // subscriptions
    // ----------------------------
    sub_currentP_ =
      create_subscription<std_msgs::msg::Float64MultiArray>(
        "/ur10skku/currentP", 10,
        std::bind(&VrCalibration::cbCurrentP, this, std::placeholders::_1));

    sub_vr_ =
      create_subscription<geometry_msgs::msg::PoseStamped>(
        "/raw_pose", 10,
        std::bind(&VrCalibration::cbVR, this, std::placeholders::_1));

    // for T_SA update
    sub_calibrated_pose_ =
      create_subscription<std_msgs::msg::Float64MultiArray>(
        "/calibrated_pose", 10,
        std::bind(&VrCalibration::cbCalibratedPose, this, std::placeholders::_1));

    // ----------------------------
    // yaml path
    // ----------------------------
    const char* home = std::getenv("HOME");
    if (!home) throw std::runtime_error("HOME env not set");
    calib_yaml_path_ = std::string(home) + "/nrs_act/behavior_ws/src/vive_tracker_ros2/yaml/calibration_matrix.yaml";

    // load existing constants (T_CE, T_SA_old)
    loadExistingYamlConstants();

    RCLCPP_INFO(get_logger(),
      "Loaded %zu waypoints (%zu target points, flag!=0). Auto-capture enabled.",
      waypoints_.size(), target_indices_.size());

    RCLCPP_INFO(get_logger(),
      "[T_SA_MODE] t_sa_mode=%s, t_sa_max_delta_deg=%.1f",
      t_sa_mode_.c_str(), t_sa_max_delta_deg_);
  }

  void run()
  {
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(shared_from_this());

    if (target_indices_.empty()) {
      RCLCPP_WARN(get_logger(), "No target points (flag!=0). Nothing to capture.");
      return;
    }

    // ==========================================================
    // (0) Pre-phase: compute T_SA once BEFORE capture starts
    // ==========================================================
    computeTSAOnceBeforeCapture(exec);

    // Now start capture loop
    enum class State { WAIT_ENTER, IN_REGION };
    State state = State::WAIT_ENTER;

    size_t target_k = 0;
    rclcpp::Time target_start_time = tnow();

    resetMotionDetector();
    bool hold_active = false;
    rclcpp::Time hold_start_time = tnow();

    rclcpp::Rate rate(loop_hz_);

    while (rclcpp::ok() && target_k < target_indices_.size()) {
      exec.spin_some();

      std::array<double,6> cp;
      std::array<double,7> vr;
      rclcpp::Time cp_t, vr_t;
      uint64_t cp_seq = 0;

      if (!getLatestData(cp, vr, cp_t, vr_t, cp_seq)) {
        rate.sleep();
        continue;
      }
      if (!isCpFresh(cp_t)) {
        rate.sleep();
        continue;
      }

      const size_t wp_idx = target_indices_[target_k];
      const auto& target_pose = waypoints_[wp_idx].pose;

      if ((tnow() - target_start_time).seconds() > target_timeout_s_) {
        RCLCPP_WARN(get_logger(),
          "[TIMEOUT] target %zu/%zu (wp line %zu). Skipping.",
          target_k+1, target_indices_.size(), wp_idx+1);
        target_k++;
        state = State::WAIT_ENTER;
        target_start_time = tnow();
        resetMotionDetector();
        hold_active = false;
        rate.sleep();
        continue;
      }

      const double dist_mm = posDistMm(cp, target_pose);
      const double ang_deg = oriErrDeg(cp, target_pose);

      if (state == State::WAIT_ENTER) {
        if (dist_mm <= pos_enter_mm_ && ang_deg <= ori_enter_deg_) {
          state = State::IN_REGION;
          hold_active = false;
          resetMotionDetector();
          target_start_time = tnow();

          RCLCPP_INFO(get_logger(),
            "[IN] target %zu/%zu (wp line %zu) | dist=%.2fmm ang=%.2fdeg",
            target_k+1, target_indices_.size(), wp_idx+1, dist_mm, ang_deg);
        }
        rate.sleep();
        continue;
      }

      if (dist_mm >= pos_exit_mm_ || ang_deg >= ori_exit_deg_) {
        state = State::WAIT_ENTER;
        hold_active = false;
        resetMotionDetector();
        RCLCPP_WARN(get_logger(),
          "[OUT] left region -> WAIT_ENTER | dist=%.2fmm ang=%.2fdeg",
          dist_mm, ang_deg);
        rate.sleep();
        continue;
      }

      if (!(dist_mm <= pos_enter_mm_ && ang_deg <= ori_enter_deg_)) {
        hold_active = false;
      }

      updateMotionIfNew(cp, cp_t, cp_seq);

      const bool stopped_now = isStoppedNow();
      if (!(dist_mm <= pos_enter_mm_ && ang_deg <= ori_enter_deg_)) {
        hold_active = false;
      } else if (!stopped_now) {
        hold_active = false;
      } else {
        if (!hold_active) {
          hold_active = true;
          hold_start_time = tnow();
        }
      }

      if (hold_active) {
        const double held = (tnow() - hold_start_time).seconds();
        if (held >= hold_time_s_) {

          if (!isVrFreshForCapture(vr_t)) {
            RCLCPP_WARN_THROTTLE(
              get_logger(), steady_clock_, 2000,
              "[WAIT_VR] VR too old (age=%.2fs). Waiting...",
              (tnow() - vr_t).seconds());
            rate.sleep();
            continue;
          }

          captureOnce(target_k, wp_idx, cp, vr, dist_mm, ang_deg);

          target_k++;
          state = State::WAIT_ENTER;
          target_start_time = tnow();
          resetMotionDetector();
          hold_active = false;
        }
      }

      rate.sleep();
    }

    RCLCPP_INFO(get_logger(), "All target waypoints processed.");

    // ==========================================================
    // (final) compute T_BC / T_AD_avg and write YAML ONCE with:
    //         R_Adj, T_AD, T_BC, T_SA, T_CE
    // ==========================================================
    try {
      finalizeCalibrationAndSaveYaml();
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "finalizeCalibrationAndSaveYaml() failed: %s", e.what());
    }
  }

private:
  // ---------- clocks ----------
  mutable rclcpp::Clock steady_clock_;
  rclcpp::Time tnow() const { return steady_clock_.now(); }

  // ---------- capture params ----------
  double pos_enter_mm_{20.0};
  double pos_exit_mm_{60.0};
  double ori_enter_deg_{25.0};
  double ori_exit_deg_{60.0};
  double vel_thresh_mms_{15.0};
  double angvel_thresh_dps_{8.0};
  double hold_time_s_{0.35};
  double cp_fresh_s_{1.0};
  double vr_capture_age_s_{30.0};
  double target_timeout_s_{300.0};
  double loop_hz_{200.0};
  size_t cp_unit_probe_N_{30};

  // ---------- T_SA params ----------
  double t_sa_w_des_z_{1.5707963267948966};
  double t_sa_wait_timeout_s_{15.0};
  double t_sa_hold_s_{0.25};
  double t_sa_fresh_s_{1.0};

  // ✅ NEW
  std::string t_sa_mode_{"keep"};     // keep/update
  double t_sa_max_delta_deg_{20.0};   // update guard

  // ---------- units ----------
  bool wp_rotvec_in_degrees_{false};
  bool cp_rotvec_unit_decided_{false};
  bool cp_rotvec_in_degrees_{false};
  size_t cp_probe_cnt_{0};
  double cp_probe_max_abs_{0.0};

  // VR position unit auto-detect (meters vs mm)
  bool vr_pos_unit_decided_{false};
  bool vr_pos_in_mm_{false};

  // ---------- waypoints ----------
  std::vector<Waypoint> waypoints_;
  std::vector<size_t> target_indices_;

  // ---------- latest data ----------
  std::mutex mtx_;
  std::array<double,6> last_cp_{};
  std::array<double,7> last_vr_{};
  bool have_cp_{false};
  bool have_vr_{false};
  rclcpp::Time last_cp_time_;
  rclcpp::Time last_vr_time_;
  uint64_t cp_seq_{0};

  // calibrated_pose (for T_SA)
  bool have_cal_pose_{false};
  std::array<double,6> last_cal_pose_{};
  rclcpp::Time last_cal_pose_time_;

  // ---------- motion detector ----------
  bool have_prev_motion_{false};
  std::array<double,6> prev_motion_cp_{};
  rclcpp::Time prev_motion_time_;
  uint64_t prev_motion_seq_{0};
  double last_vnorm_mms_{1e9};
  double last_omega_dps_{1e9};

  // ---------- ROS ----------
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_currentP_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_vr_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_calibrated_pose_;

  // ---------- files ----------
  std::string waypoint_file_;
  std::string ee_path_, vr_path_;
  std::ofstream ee_ofs_, vr_ofs_;

  // ---------- YAML / calibration outputs ----------
  std::string calib_yaml_path_;

  // existing constants from yaml
  Eigen::Matrix4d T_CE_ = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_SA_old_ = Eigen::Matrix4d::Identity();

  // computed outputs
  bool have_radj_{false};
  Eigen::Matrix3d R_adj_ = Eigen::Matrix3d::Identity();

  bool t_sa_computed_{false};
  Eigen::Matrix4d T_SA_new_ = Eigen::Matrix4d::Identity();

  // store the first two captured VR poses (for R_Adj)
  bool have_pose1_{false}, have_pose2_{false};
  Eigen::Quaterniond q_pose1_{1,0,0,0}; // (w,x,y,z)
  Eigen::Quaterniond q_pose2_{1,0,0,0};

  // store all captured samples for T_BC / T_AD
  std::vector<Eigen::Matrix4d> T_AB_all_; // arm
  std::vector<Eigen::Matrix4d> T_DC_all_; // tracker

  // storage for finalize step
  std::vector<Eigen::Vector3d> O_B0B1_list_;
  std::vector<Eigen::Vector3d> O_C0C1_list_;
  std::vector<Eigen::Matrix4d> T_AB0_list_;
  std::vector<Eigen::Matrix4d> T_DC0_list_;

  // ---------- callbacks ----------
  void cbCurrentP(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (msg->data.size() < 6) return;
    const auto ts = tnow();

    std::lock_guard<std::mutex> lk(mtx_);
    for (int i=0;i<6;i++) last_cp_[i] = msg->data[i];
    have_cp_ = true;
    last_cp_time_ = ts;
    cp_seq_++;

    // auto-detect unit of currentP rotvec
    if (!cp_rotvec_unit_decided_) {
      cp_probe_cnt_++;
      cp_probe_max_abs_ = std::max(cp_probe_max_abs_, std::fabs(last_cp_[3]));
      cp_probe_max_abs_ = std::max(cp_probe_max_abs_, std::fabs(last_cp_[4]));
      cp_probe_max_abs_ = std::max(cp_probe_max_abs_, std::fabs(last_cp_[5]));

      if (cp_probe_cnt_ >= cp_unit_probe_N_) {
        cp_rotvec_in_degrees_ = (cp_probe_max_abs_ > 6.0);
        cp_rotvec_unit_decided_ = true;
        RCLCPP_INFO(get_logger(),
          "currentP rotvec unit decided: %s (max_abs=%.3f over %zu samples)",
          cp_rotvec_in_degrees_ ? "DEG" : "RAD", cp_probe_max_abs_, cp_probe_cnt_);
      }
    }
  }

  void cbVR(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    const auto ts = tnow();
    std::lock_guard<std::mutex> lk(mtx_);
    last_vr_[0]=msg->pose.position.x;
    last_vr_[1]=msg->pose.position.y;
    last_vr_[2]=msg->pose.position.z;
    last_vr_[3]=msg->pose.orientation.x;
    last_vr_[4]=msg->pose.orientation.y;
    last_vr_[5]=msg->pose.orientation.z;
    last_vr_[6]=msg->pose.orientation.w;
    have_vr_ = true;
    last_vr_time_ = ts;

    if (!vr_pos_unit_decided_) {
      double mabs = 0.0;
      mabs = std::max(mabs, std::fabs(last_vr_[0]));
      mabs = std::max(mabs, std::fabs(last_vr_[1]));
      mabs = std::max(mabs, std::fabs(last_vr_[2]));
      vr_pos_in_mm_ = (mabs > 10.0);
      vr_pos_unit_decided_ = true;
      RCLCPP_INFO(get_logger(),
        "raw_pose position unit decided (heuristic): %s (max_abs=%.3f)",
        vr_pos_in_mm_ ? "MM" : "M", mabs);
    }
  }

  void cbCalibratedPose(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (msg->data.size() < 6) return;
    const auto ts = tnow();
    std::lock_guard<std::mutex> lk(mtx_);
    for (int i=0;i<6;i++) last_cal_pose_[i] = msg->data[i];
    last_cal_pose_time_ = ts;
    have_cal_pose_ = true;
  }

  // ---------- waypoint parsing ----------
  void loadWaypointsAndDecideWpUnits()
  {
    std::ifstream ifs(waypoint_file_);
    if (!ifs.is_open())
      throw std::runtime_error("Failed to load waypoint file: " + waypoint_file_);

    double max_abs_w = 0.0;
    std::string line;

    while (std::getline(ifs, line)) {
      if (line.empty()) continue;

      std::stringstream ss(line);
      Waypoint wp;

      for (int i=0;i<6;i++) {
        if (!(ss >> wp.pose[i]))
          throw std::runtime_error("Waypoint parse error in line: " + line);
      }

      double dummy1, dummy2;
      if (!(ss >> dummy1 >> dummy2 >> wp.flag))
        throw std::runtime_error("Waypoint tail parse error in line: " + line);

      max_abs_w = std::max(max_abs_w, std::fabs(wp.pose[3]));
      max_abs_w = std::max(max_abs_w, std::fabs(wp.pose[4]));
      max_abs_w = std::max(max_abs_w, std::fabs(wp.pose[5]));

      waypoints_.push_back(wp);
    }

    wp_rotvec_in_degrees_ = (max_abs_w > 6.0);
  }

  void buildTargetIndices()
  {
    target_indices_.clear();
    for (size_t i=0; i<waypoints_.size(); ++i) {
      if (std::fabs(waypoints_[i].flag) > 1e-12)
        target_indices_.push_back(i);
    }
  }

  // ---------- latest data fetch ----------
  bool getLatestData(std::array<double,6>& cp,
                     std::array<double,7>& vr,
                     rclcpp::Time& cp_t,
                     rclcpp::Time& vr_t,
                     uint64_t& cp_seq_out)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!have_cp_) return false;

    cp = last_cp_;
    cp_t = last_cp_time_;
    cp_seq_out = cp_seq_;

    if (have_vr_) {
      vr = last_vr_;
      vr_t = last_vr_time_;
    } else {
      vr_t = rclcpp::Time(0,0,RCL_STEADY_TIME);
    }
    return true;
  }

  bool isCpFresh(const rclcpp::Time& cp_t) const
  {
    return (tnow() - cp_t).seconds() <= cp_fresh_s_;
  }

  bool isVrFreshForCapture(const rclcpp::Time& vr_t) const
  {
    if (!have_vr_) return false;
    return (tnow() - vr_t).seconds() <= vr_capture_age_s_;
  }

  bool isCalPoseFresh() const
  {
    if (!have_cal_pose_) return false;
    return (tnow() - last_cal_pose_time_).seconds() <= t_sa_fresh_s_;
  }

  // ---------- reach metrics ----------
  static double posDistMm(const std::array<double,6>& cp,
                          const std::array<double,6>& target)
  {
    double e2 = 0.0;
    for (int i=0;i<3;i++){
      double d = cp[i] - target[i];
      e2 += d*d;
    }
    return std::sqrt(e2);
  }

  std::array<double,3> toRotvecRad_WP(const std::array<double,6>& pose) const
  {
    std::array<double,3> w = {pose[3], pose[4], pose[5]};
    if (wp_rotvec_in_degrees_) {
      w[0] = deg2rad(w[0]);
      w[1] = deg2rad(w[1]);
      w[2] = deg2rad(w[2]);
    }
    return w;
  }

  std::array<double,3> toRotvecRad_CP(const std::array<double,6>& pose) const
  {
    std::array<double,3> w = {pose[3], pose[4], pose[5]};
    bool cp_deg = cp_rotvec_unit_decided_ ? cp_rotvec_in_degrees_ : false;
    if (cp_deg) {
      w[0] = deg2rad(w[0]);
      w[1] = deg2rad(w[1]);
      w[2] = deg2rad(w[2]);
    }
    return w;
  }

  double oriErrDeg(const std::array<double,6>& cp,
                   const std::array<double,6>& target) const
  {
    const auto w_c = toRotvecRad_CP(cp);
    const auto w_t = toRotvecRad_WP(target);

    std::array<double,9> Rc, Rt;
    rotvecToRotMatRad(w_c, Rc);
    rotvecToRotMatRad(w_t, Rt);

    const double ang_rad = rotAngleBetweenRad(Rt, Rc);
    return rad2deg(ang_rad);
  }

  // ---------- motion detector ----------
  void resetMotionDetector()
  {
    have_prev_motion_ = false;
    prev_motion_seq_  = 0;
    last_vnorm_mms_   = 1e9;
    last_omega_dps_   = 1e9;
  }

  void updateMotionIfNew(const std::array<double,6>& cp,
                         const rclcpp::Time& cp_time,
                         uint64_t cp_seq_in)
  {
    if (cp_seq_in == prev_motion_seq_) return;

    if (!have_prev_motion_) {
      prev_motion_cp_   = cp;
      prev_motion_time_ = cp_time;
      have_prev_motion_ = true;
      prev_motion_seq_  = cp_seq_in;
      return;
    }

    const double dt = (cp_time - prev_motion_time_).seconds();
    if (dt <= 1e-4) {
      prev_motion_cp_   = cp;
      prev_motion_time_ = cp_time;
      prev_motion_seq_  = cp_seq_in;
      return;
    }

    double v2 = 0.0;
    for (int i=0;i<3;i++){
      double v = (cp[i] - prev_motion_cp_[i]) / dt;
      v2 += v*v;
    }
    last_vnorm_mms_ = std::sqrt(v2);

    const auto w_prev = toRotvecRad_CP(prev_motion_cp_);
    const auto w_cur  = toRotvecRad_CP(cp);

    std::array<double,9> Rprev, Rcur;
    rotvecToRotMatRad(w_prev, Rprev);
    rotvecToRotMatRad(w_cur,  Rcur);

    const double dang_rad = rotAngleBetweenRad(Rprev, Rcur);
    last_omega_dps_ = rad2deg(dang_rad) / dt;

    prev_motion_cp_   = cp;
    prev_motion_time_ = cp_time;
    prev_motion_seq_  = cp_seq_in;
  }

  bool isStoppedNow() const
  {
    return (last_vnorm_mms_ <= vel_thresh_mms_) && (last_omega_dps_ <= angvel_thresh_dps_);
  }

  // ==========================================================
  // T_SA computation (right-multiply)
  // ==========================================================
  bool computeTSAFromLatestCalPose()
  {
    if (!have_cal_pose_ || !isCalPoseFresh()) return false;

    // w_meas (rad) from /calibrated_pose: [x y z wx wy wz]
    std::array<double,3> w_meas = { last_cal_pose_[3], last_cal_pose_[4], last_cal_pose_[5] };

    // R_total from w_meas
    std::array<double,9> Rtot_arr;
    rotvecToRotMatRad(w_meas, Rtot_arr);
    Eigen::Matrix3d R_total;
    R_total << Rtot_arr[0], Rtot_arr[1], Rtot_arr[2],
               Rtot_arr[3], Rtot_arr[4], Rtot_arr[5],
               Rtot_arr[6], Rtot_arr[7], Rtot_arr[8];

    // desired R_des from w_des=[0,0,t_sa_w_des_z_]
    std::array<double,3> w_des = {0.0, 0.0, t_sa_w_des_z_};
    std::array<double,9> Rdes_arr;
    rotvecToRotMatRad(w_des, Rdes_arr);
    Eigen::Matrix3d R_des;
    R_des << Rdes_arr[0], Rdes_arr[1], Rdes_arr[2],
             Rdes_arr[3], Rdes_arr[4], Rdes_arr[5],
             Rdes_arr[6], Rdes_arr[7], Rdes_arr[8];

    const Eigen::Matrix3d R_SA_old = T_SA_old_.block<3,3>(0,0);
    const Eigen::Matrix3d R_SA_new = R_SA_old * R_total.transpose() * R_des;

    // ✅ guard: old->new 변화가 너무 크면 reject
    const double delta_deg = rotDiffAngleDeg(R_SA_old, R_SA_new);
    if (delta_deg > t_sa_max_delta_deg_) {
      RCLCPP_WARN(get_logger(),
        "[T_SA_REJECT] delta too large: %.2f deg (limit=%.2f deg). Keeping old T_SA.",
        delta_deg, t_sa_max_delta_deg_);
      return false;
    }

    T_SA_new_ = Eigen::Matrix4d::Identity();
    T_SA_new_.block<3,3>(0,0) = R_SA_new;

    t_sa_computed_ = true;

    RCLCPP_INFO(get_logger(),
      "[T_SA_DONE] Computed T_SA (right-multiply). delta=%.2fdeg, w_meas(rad)=[%.6f %.6f %.6f], w_des=[0 0 %.6f]",
      delta_deg, w_meas[0], w_meas[1], w_meas[2], t_sa_w_des_z_);

    return true;
  }

  void computeTSAOnceBeforeCapture(rclcpp::executors::SingleThreadedExecutor& exec)
  {
    // ✅ keep 모드면: 기존 YAML의 T_SA를 그대로 유지
    if (t_sa_mode_ == "keep") {
      t_sa_computed_ = false;          // "새로 계산"은 안 함
      T_SA_new_ = Eigen::Matrix4d::Identity();
      RCLCPP_INFO(get_logger(), "[T_SA] mode=keep: will keep existing T_SA from yaml (no recompute).");
      return;
    }

    // update 모드
    RCLCPP_INFO(get_logger(),
      "[T_SA] mode=update: waiting for /calibrated_pose (fresh<=%.2fs) and robot STOP hold(%.2fs), timeout=%.1fs",
      t_sa_fresh_s_, t_sa_hold_s_, t_sa_wait_timeout_s_);

    const rclcpp::Time t0 = tnow();
    bool hold_active = false;
    rclcpp::Time hold_start = tnow();

    resetMotionDetector();

    rclcpp::Rate rate(std::min(200.0, loop_hz_));

    while (rclcpp::ok() && (tnow() - t0).seconds() < t_sa_wait_timeout_s_) {
      exec.spin_some();

      std::array<double,6> cp;
      std::array<double,7> vr;
      rclcpp::Time cp_t, vr_t;
      uint64_t cp_seq = 0;

      if (!getLatestData(cp, vr, cp_t, vr_t, cp_seq)) {
        rate.sleep();
        continue;
      }
      if (!isCpFresh(cp_t)) {
        rate.sleep();
        continue;
      }

      updateMotionIfNew(cp, cp_t, cp_seq);

      const bool stopped_now = isStoppedNow();
      if (!stopped_now) {
        hold_active = false;
        rate.sleep();
        continue;
      }

      if (!isCalPoseFresh()) {
        hold_active = false;
        rate.sleep();
        continue;
      }

      if (!hold_active) {
        hold_active = true;
        hold_start = tnow();
      }

      const double held = (tnow() - hold_start).seconds();
      if (held >= t_sa_hold_s_) {
        if (computeTSAFromLatestCalPose()) {
          RCLCPP_INFO(get_logger(), "[T_SA] Pre-capture update done.");
          return;
        } else {
          // reject 등으로 실패하면 계속 기다려서 다시 시도
        }
      }

      rate.sleep();
    }

    // timeout: update 실패 -> old 유지
    RCLCPP_WARN(get_logger(),
      "[T_SA_WARN] Pre-capture T_SA update timeout or rejected. Will KEEP existing T_SA from yaml.");
    t_sa_computed_ = false;
  }

  // ---------- capture ----------
  void captureOnce(size_t target_k, size_t wp_idx,
                   const std::array<double,6>& cp,
                   const std::array<double,7>& vr,
                   double dist_mm, double ang_deg)
  {
    // --- EE rotation from rotvec ---
    const auto w_c = toRotvecRad_CP(cp);
    std::array<double,9> Rarr;
    rotvecToRotMatRad(w_c, Rarr);

    // --- write files (meters) ---
    const double cp_x_m = cp[0] * 1e-3;
    const double cp_y_m = cp[1] * 1e-3;
    const double cp_z_m = cp[2] * 1e-3;

    double vr_x = vr[0], vr_y = vr[1], vr_z = vr[2];
    if (vr_pos_unit_decided_ && vr_pos_in_mm_) {
      vr_x *= 1e-3; vr_y *= 1e-3; vr_z *= 1e-3;
    }

    ee_ofs_
      << Rarr[0]<<" "<<Rarr[1]<<" "<<Rarr[2]<<" "<<cp_x_m<<" "
      << Rarr[3]<<" "<<Rarr[4]<<" "<<Rarr[5]<<" "<<cp_y_m<<" "
      << Rarr[6]<<" "<<Rarr[7]<<" "<<Rarr[8]<<" "<<cp_z_m<<"\n";

    vr_ofs_
      << vr_x<<" "<<vr_y<<" "<<vr_z<<" "
      << vr[3]<<" "<<vr[4]<<" "<<vr[5]<<" "<<vr[6]<<"\n";

    ee_ofs_.flush();
    vr_ofs_.flush();

    // --- store samples for calibration (meters) ---
    Eigen::Matrix3d R_ab;
    R_ab << Rarr[0], Rarr[1], Rarr[2],
            Rarr[3], Rarr[4], Rarr[5],
            Rarr[6], Rarr[7], Rarr[8];
    Eigen::Vector3d p_ab(cp_x_m, cp_y_m, cp_z_m);
    T_AB_all_.push_back(makeT(R_ab, p_ab));

    // VR transform from quaternion (ROS order: x y z w)
    Eigen::Quaterniond q_vr(vr[6], vr[3], vr[4], vr[5]); // ctor: (w,x,y,z)
    q_vr.normalize();
    Eigen::Matrix3d R_dc = q_vr.toRotationMatrix();
    Eigen::Vector3d p_dc(vr_x, vr_y, vr_z);
    T_DC_all_.push_back(makeT(R_dc, p_dc));

    // --- R_Adj from the first two captured VR poses ---
    if (target_k == 0 && !have_pose1_) {
      q_pose1_ = q_vr;
      have_pose1_ = true;
      RCLCPP_INFO(get_logger(), "[R_ADJ] pose1 stored (from 1st capture).");
    } else if (target_k == 1 && !have_pose2_) {
      q_pose2_ = q_vr;
      have_pose2_ = true;

      Eigen::Matrix3d R1 = q_pose1_.toRotationMatrix();
      Eigen::Matrix3d R2 = q_pose2_.toRotationMatrix();
      R_adj_ = R1.transpose() * R2;
      have_radj_ = true;

      RCLCPP_INFO(get_logger(),
        "[R_ADJ_DONE] R_Adj computed after 2nd capture (wp line %zu).\n"
        "R_Adj=\n"
        "[% .6f % .6f % .6f]\n"
        "[% .6f % .6f % .6f]\n"
        "[% .6f % .6f % .6f]",
        wp_idx + 1,
        R_adj_(0,0), R_adj_(0,1), R_adj_(0,2),
        R_adj_(1,0), R_adj_(1,1), R_adj_(1,2),
        R_adj_(2,0), R_adj_(2,1), R_adj_(2,2));
    }

    RCLCPP_INFO(get_logger(),
      "[CAPTURE] target %zu/%zu (wp line %zu) | dist=%.2fmm ang=%.2fdeg | v=%.2fmm/s w=%.2fdeg/s",
      target_k+1, target_indices_.size(), wp_idx+1,
      dist_mm, ang_deg, last_vnorm_mms_, last_omega_dps_);
  }

  // ---------- YAML read (constants) ----------
  void loadExistingYamlConstants()
  {
    // defaults
    T_CE_ = Eigen::Matrix4d::Identity();
    T_CE_(2,3) = 0.222;

    T_SA_old_ = Eigen::Matrix4d::Identity();

    try {
      YAML::Node existing = YAML::LoadFile(calib_yaml_path_);

      if (existing["T_CE"]) {
        Eigen::Matrix4d tmp = Eigen::Matrix4d::Identity();
        if (readMat4(existing["T_CE"], tmp)) {
          T_CE_ = tmp;
          RCLCPP_INFO(get_logger(), "[YAML] Loaded existing T_CE.");
        }
      }

      if (existing["T_SA"]) {
        Eigen::Matrix4d tmp = Eigen::Matrix4d::Identity();
        if (readMat4(existing["T_SA"], tmp)) {
          T_SA_old_ = tmp;
          RCLCPP_INFO(get_logger(), "[YAML] Loaded existing T_SA (old).");
        }
      } else {
        RCLCPP_WARN(get_logger(), "[YAML] T_SA not found. Assume Identity (old).");
      }

    } catch (...) {
      RCLCPP_WARN(get_logger(), "[YAML] Cannot load existing yaml. Will create new: %s", calib_yaml_path_.c_str());
    }
  }

  bool readMat4(const YAML::Node& n, Eigen::Matrix4d& T)
  {
    if (!n || !n.IsSequence() || n.size() != 4) return false;
    for (int r=0;r<4;r++){
      if (!n[r].IsSequence() || n[r].size() != 4) return false;
      for (int c=0;c<4;c++){
        T(r,c) = n[r][c].as<double>();
      }
    }
    return true;
  }

  bool readMat3(const YAML::Node& n, Eigen::Matrix3d& R)
  {
    if (!n || !n.IsSequence() || n.size() != 3) return false;
    for (int r=0;r<3;r++){
      if (!n[r].IsSequence() || n[r].size() != 3) return false;
      for (int c=0;c<3;c++){
        R(r,c) = n[r][c].as<double>();
      }
    }
    return true;
  }

  // ---------- compute T_BC / T_AD_avg and save yaml ----------
  void finalizeCalibrationAndSaveYaml()
  {
    const size_t N = T_AB_all_.size();
    if (N < 2 || T_DC_all_.size() != N) {
      throw std::runtime_error("Not enough samples to compute calibration (need >=2).");
    }

    // clear buffers
    O_B0B1_list_.clear();
    O_C0C1_list_.clear();
    T_AB0_list_.clear();
    T_DC0_list_.clear();

    const size_t K = N - 1;

    Eigen::MatrixXd M(9 * K, 9);
    Eigen::MatrixXd K1(3 * K, 3);
    Eigen::VectorXd K2(3 * K);

    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    for (size_t i=0; i<K; i++) {
      const Eigen::Matrix4d& T_AB0 = T_AB_all_[i];
      const Eigen::Matrix4d& T_AB1 = T_AB_all_[i+1];

      const Eigen::Matrix4d& T_DC0 = T_DC_all_[i];
      const Eigen::Matrix4d& T_DC1 = T_DC_all_[i+1];

      const Eigen::Matrix4d T_B0B1 = invT(T_AB0) * T_AB1;
      const Eigen::Matrix4d T_C0C1 = invT(T_DC0) * T_DC1;

      const Eigen::Matrix3d R_B0B1 = T_B0B1.block<3,3>(0,0);
      const Eigen::Vector3d O_B0B1 = T_B0B1.block<3,1>(0,3);

      const Eigen::Matrix3d R_C0C1 = T_C0C1.block<3,3>(0,0);
      const Eigen::Vector3d O_C0C1 = T_C0C1.block<3,1>(0,3);

      Eigen::Matrix<double,9,9> m = kron3(I, R_B0B1) - kron3(R_C0C1.transpose(), I);
      M.block(9*i, 0, 9, 9) = m;

      K1.block(3*i, 0, 3, 3) = (I - R_B0B1);

      O_B0B1_list_.push_back(O_B0B1);
      O_C0C1_list_.push_back(O_C0C1);
      T_AB0_list_.push_back(T_AB0);
      T_DC0_list_.push_back(T_DC0);
    }

    Eigen::MatrixXd X = M.transpose() * M;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(X);
    if (es.info() != Eigen::Success) throw std::runtime_error("EigenSolver failed on X=M^T*M");

    Eigen::VectorXd vectX = es.eigenvectors().col(0);
    Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::ColMajor>> R_BC_raw(vectX.data());

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_BC_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R_BC = U * V.transpose();
    if (R_BC.determinant() < 0) {
      U.col(2) *= -1.0;
      R_BC = U * V.transpose();
    }

    for (size_t i=0; i<K; i++) {
      const Eigen::Vector3d& O_B0B1 = O_B0B1_list_[i];
      const Eigen::Vector3d& O_C0C1 = O_C0C1_list_[i];
      Eigen::Vector3d temp = O_B0B1 - R_BC * O_C0C1;
      K2.segment<3>(3*i) = temp;
    }

    Eigen::Vector3d O_BC = K1.colPivHouseholderQr().solve(K2);
    Eigen::Matrix4d T_BC = makeT(R_BC, O_BC);

    // Compute T_AD_i = T_AB0_i * T_BC * inv(T_DC0_i)
    std::vector<Eigen::Quaterniond> quats;
    quats.reserve(K);
    Eigen::Vector3d t_sum = Eigen::Vector3d::Zero();

    for (size_t i=0; i<K; i++) {
      const Eigen::Matrix4d& T_AB0 = T_AB0_list_[i];
      const Eigen::Matrix4d& T_DC0 = T_DC0_list_[i];

      Eigen::Matrix4d T_AD = T_AB0 * T_BC * invT(T_DC0);
      Eigen::Matrix3d R = T_AD.block<3,3>(0,0);
      Eigen::Vector3d t = T_AD.block<3,1>(0,3);

      Eigen::Quaterniond q(R);
      q.normalize();
      quats.push_back(q);
      t_sum += t;
    }

    Eigen::Quaterniond q_ref = quats.front();
    Eigen::Vector4d q_sum = Eigen::Vector4d::Zero();
    for (auto& q : quats) {
      if (q_ref.coeffs().dot(q.coeffs()) < 0) {
        q.coeffs() *= -1.0;
      }
      q_sum += q.coeffs();
    }
    q_sum /= static_cast<double>(quats.size());
    Eigen::Quaterniond q_mean;
    q_mean.coeffs() = q_sum;
    q_mean.normalize();

    Eigen::Vector3d t_mean = t_sum / static_cast<double>(K);

    Eigen::Matrix4d T_AD_avg = makeT(q_mean.toRotationMatrix(), t_mean);

    // Decide final T_SA to save:
    // - keep mode: always save old
    // - update mode: if computed => save new else save old
    Eigen::Matrix4d T_SA_to_save = T_SA_old_;
    if (t_sa_mode_ == "update") {
      if (t_sa_computed_) T_SA_to_save = T_SA_new_;
      else T_SA_to_save = T_SA_old_;
    }

    writeCalibrationYamlAll(
      T_AD_avg,
      T_BC,
      have_radj_ ? R_adj_ : Eigen::Matrix3d::Identity(),
      T_CE_,
      T_SA_to_save
    );

    RCLCPP_INFO(get_logger(),
      "[YAML_SAVED] T_AD, T_BC computed. R_Adj=%s. T_SA saved by mode=%s (%s). -> %s",
      have_radj_ ? "computed" : "IDENTITY(fallback)",
      t_sa_mode_.c_str(),
      (t_sa_mode_=="update" ? (t_sa_computed_ ? "new(computed)" : "old(fallback)") : "old(kept)"),
      calib_yaml_path_.c_str());
  }

  void writeCalibrationYamlAll(const Eigen::Matrix4d& T_AD,
                               const Eigen::Matrix4d& T_BC,
                               const Eigen::Matrix3d& R_Adj,
                               const Eigen::Matrix4d& T_CE,
                               const Eigen::Matrix4d& T_SA)
  {
    std::ofstream ofs(calib_yaml_path_, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) throw std::runtime_error("Failed to open yaml: " + calib_yaml_path_);

    const int prec = 12;

    ofs << "# VR calibration matrix setting\n";
    ofs << "# Auto-capture + One-shot YAML update (R_Adj, T_AD, T_BC, T_SA)\n";
    ofs << "# saved_at: " << nowLocalString() << "\n\n";

    ofs << "meta:\n";
    ofs << "  t_sa_w_des_z: " << std::fixed << std::setprecision(prec) << t_sa_w_des_z_ << "\n";
    ofs << "  note: \"T_SA is right-multiplied in vive_tracker (M_cal = ... @ T_SA)\"\n\n";

    auto writeMat4 = [&](const std::string& key, const Eigen::Matrix4d& T){
      ofs << key << ":\n";
      ofs << std::fixed << std::setprecision(prec);
      for (int r=0;r<4;r++){
        ofs << "  - [";
        for (int c=0;c<4;c++){
          ofs << T(r,c);
          if (c<3) ofs << ", ";
        }
        ofs << "]\n";
      }
      ofs << "\n";
    };

    auto writeMat3 = [&](const std::string& key, const Eigen::Matrix3d& R){
      ofs << key << ":\n";
      ofs << std::fixed << std::setprecision(prec);
      for (int r=0;r<3;r++){
        ofs << "  - [";
        for (int c=0;c<3;c++){
          ofs << R(r,c);
          if (c<2) ofs << ", ";
        }
        ofs << "]\n";
      }
      ofs << "\n";
    };

    writeMat4("T_AD", T_AD);
    writeMat4("T_BC", T_BC);
    writeMat3("R_Adj", R_Adj);

    ofs << "# constant offset: tune here if needed\n";
    writeMat4("T_CE", T_CE);

    ofs << "# spatial-angle frame alignment (right-multiply)\n";
    writeMat4("T_SA", T_SA);

    ofs.flush();
  }
};

// ================= main =================
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<VrCalibration>();
    node->run();
  } catch (const std::exception& e) {
    std::cerr << "vr_calibration exception: " << e.what() << std::endl;
  }
  rclcpp::shutdown();
  return 0;
}
