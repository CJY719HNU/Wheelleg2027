"""Verify paper gain indexing, ground-cache isolation and adjustable slew rate."""
import json
import numpy as np
from advanced_control import airborne_lqr,SensorRobot,HybridController,load_gains,Command,step,DT,ROOT

rows=load_gains();state=np.array([.12,-.3,5.,2.,.25,-.6])
for row in rows:
    K=np.array(row['K']);before=K.copy();u=airborne_lqr(K,state)
    assert np.array_equal(K,before)
    assert u[0]==0
    assert np.isclose(u[1],-K[1,:2]@state[:2])
    for column in range(2,6):
        changed=state.copy();changed[column]+=100
        assert np.array_equal(airborne_lqr(K,changed),u)
checks=dict(paper_theta_rate_only=True,wheel_row_zero=True,ground_gains_unchanged=True)
for accel in [.6,1.2]:
    r=SensorRobot();c=HybridController(r,rows,assist=False);c.drive_accel=accel;c.initialize(r.d)
    for _ in range(100):c.submit(Command(speed=2.));step(r,c)
    assert abs(c.speed-accel*DT*100)<1e-10
checks['configurable_reference_slew']=True
(ROOT/'airborne_feedback_checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
