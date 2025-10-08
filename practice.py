import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make(
    id="FrozenLake-v1",
    render_mode="rgb_array",
    map_name='4x4',
    is_slippery=False
)

env.reset()

# 1. 무한 루프 시작
step_count = 0
while True:
    step_count += 1
    # 2. 무작위 행동 선택 (0:좌, 1:하, 2:우, 3:상)
    random_action = env.action_space.sample()
    
    # 3. 선택한 행동 실행
    obs, reward, terminated, truncated, info = env.step(random_action)
    
    print(f"{step_count}번째 시도: 행동 {random_action} -> 위치 {obs}")
    
    # 4. 게임 종료 조건 확인
    if terminated:
        print("--------------------")
        if reward == 1.0:
            print("성공! 목표 지점에 도달했습니다.")
        else:
            print("실패! 구멍에 빠졌습니다.")
        print("게임 종료!")
        break # while 루프 탈출


plt.axis('off')
plt.imshow(env.render())
plt.show()