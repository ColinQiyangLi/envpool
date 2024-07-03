/*
 * Copyright 2022 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_MUJOCO_GYM_ANTMAZE_H_
#define ENVPOOL_MUJOCO_GYM_ANTMAZE_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/gym/mujoco_env.h"

namespace mujoco_gym {

class AntMazeEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "reset_noise_scale"_.Bind(0.1), "frame_skip"_.Bind(5),
        "post_constraint"_.Bind(true));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict("obs"_.Bind(Spec<mjtNum>({29}, {-inf, inf})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 8}, {-1.0, 1.0})));
  }
};

using AntMazeEnvSpec = EnvSpec<AntMazeEnvFns>;

class AntMazeEnv : public Env<AntMazeEnvSpec>, public MujocoEnv {
 protected:
  int id_torso_;
  std::uniform_real_distribution<> dist_qpos_;
  std::normal_distribution<> dist_qvel_;

 public:
  AntMazeEnv(const Spec& spec, int env_id)
      : Env<AntMazeEnvSpec>(spec, env_id),
        MujocoEnv(spec.config["base_path"_] + "/mujoco/assets_gym/ant.xml",
                  spec.config["frame_skip"_], spec.config["post_constraint"_],
                  spec.config["max_episode_steps"_]),
        id_torso_(mj_name2id(model_, mjOBJ_XBODY, "torso")),
        dist_qpos_(-spec.config["reset_noise_scale"_],
                   spec.config["reset_noise_scale"_]),
        dist_qvel_(0, spec.config["reset_noise_scale"_]) {}

  void MujocoResetModel() override {
    for (int i = 0; i < model_->nq; ++i) {
      data_->qpos[i] = init_qpos_[i] + dist_qpos_(gen_);
    }
    for (int i = 0; i < model_->nv; ++i) {
      data_->qvel[i] = init_qvel_[i] + dist_qvel_(gen_);
    }
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    done_ = false;
    elapsed_step_ = 0;
    MujocoReset();
    WriteState(0.0);
  }

  void Step(const Action& action) override {
    // step
    mjtNum* act = static_cast<mjtNum*>(action["action"_].Data());
    MujocoStep(act);
    auto reward = static_cast<float>(0.0);
    ++elapsed_step_;
    done_ = (elapsed_step_ >= max_episode_steps_);
    WriteState(reward);
  }

 private:

  void WriteState(float reward) {
    State state = Allocate();
    state["reward"_] = reward;
    // obs
    mjtNum* obs = static_cast<mjtNum*>(state["obs"_].Data());
    for (int i = 0; i < model_->nq; ++i) {
      *(obs++) = data_->qpos[i];
    }
    for (int i = 0; i < model_->nv; ++i) {
      *(obs++) = data_->qvel[i];
    }
  }
};

using AntMazeEnvPool = AsyncEnvPool<AntMazeEnv>;

}  // namespace mujoco_gym

#endif  // ENVPOOL_MUJOCO_GYM_ANT_H_
