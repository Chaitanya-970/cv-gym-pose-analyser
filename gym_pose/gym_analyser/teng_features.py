"""Frozen Teng feature representation; silhouette is not anatomical spine truth."""

import numpy as np

def angle(a,c,d):
 u=a-c;v=d-c;den=np.linalg.norm(u)*np.linalg.norm(v)
 if den<=1e-8:raise ValueError('degenerate angle')
 return float(np.arccos(np.clip(u@v/den,-1,1))/np.pi)

def derive(guard,spine):
 if guard.get('person_count')!=1 or len(guard.get('poses',[]))!=1:raise ValueError('singleperson_unavailable')
 ps=guard['poses'][0];w,h=guard['width'],guard['height']
 def usable(ids):return all(0<=ps[i]['x_px']<w and 0<=ps[i]['y_px']<h and ps[i]['visibility']>=.5 and ps[i]['presence']>=.5 for i in ids)
 def quality(ids):return min(min(ps[i]['visibility'],ps[i]['presence']) for i in ids)
 def xy(i):return np.array([ps[i]['x_px'],ps[i]['y_px']],float)
 legs=[ids for ids in [(11,23,25,27),(12,24,26,28)] if usable(ids)];feet=[ids for ids in [(27,29,31),(28,30,32)] if usable(ids)]
 if not legs or not feet:raise ValueError('joint_or_foot_visibility_unavailable')
 leg=max(legs,key=quality);foot=max(feet,key=quality);shoulder,hip,knee,ankle=[xy(i) for i in leg];a,heel,toe=[xy(i) for i in foot];length=np.linalg.norm(heel-toe)
 if length<5:raise ValueError('foot_too_small')
 if not usable([0,23,24]):raise ValueError('facing_landmarks_unavailable')
 center=(xy(23)+xy(24))/2;nose=xy(0);direction=nose[0]-center[0]
 if abs(direction)<.05*np.linalg.norm(shoulder-hip):raise ValueError('facing_unresolved')
 posterior=1 if direction<0 else -1;features=[angle(hip,knee,ankle),angle(shoulder,hip,knee),(toe[1]-heel[1])/length,(toe[1]-a[1])/length,float(np.linalg.norm(a-heel)/length),abs(float(a[0]-toe[0]))/length]
 if spine.get('reason')!='single_person':raise ValueError('spine_unavailable')
 def chain(ids):
  if any(not spine['landmarks'][i]['usable'] for i in ids):raise ValueError('spinepoint_unusable')
  coords=[np.array([spine['landmarks'][i]['x_px'],spine['landmarks'][i]['y_px']],float) for i in ids];lo,mid,hi=coords;axis=hi-lo;den=float(np.linalg.norm(axis))
  if den<15:raise ValueError('spine_segment_too_small')
  unit=axis/den;normal=np.array([-unit[1],unit[0]]);projection=float((mid-lo)@unit/den);offset=float((mid-lo)@normal/den)
  if not .1<=projection<=.9 or abs(offset)>.6:raise ValueError('spine_geometry_unreliable')
  return [offset*(1 if normal[0]*posterior>0 else -1),projection,angle(lo,mid,hi)],den
 lumbar,ll=chain([26,27,28]);thoracic,_=chain([28,29,30]);features+=lumbar+thoracic
 lo=np.array([spine['landmarks'][19]['x_px'],spine['landmarks'][19]['y_px']],float);hi=np.array([spine['landmarks'][18]['x_px'],spine['landmarks'][18]['y_px']],float);torso=hi-lo;tl=float(np.linalg.norm(torso))
 if tl<30 or not spine['landmarks'][19]['usable'] or not spine['landmarks'][18]['usable']:raise ValueError('torso_unavailable')
 features+=[float(np.arccos(np.clip(-torso[1]/tl,-1,1))/np.pi),ll/tl];x=np.array(features,np.float32);assert x.shape==(14,) and np.isfinite(x).all();return x,dict(leg=leg,foot=foot,facing_estimate='left' if posterior==1 else 'right')

def outline(mask,low,high,posterior):
 axis=np.array(high,float)-np.array(low,float);length=float(np.linalg.norm(axis))
 if length<30:raise ValueError('short_torso_axis')
 axis/=length;normal=np.array([-axis[1],axis[0]])
 if normal[0]*posterior<0 or (abs(normal[0])<1e-8 and normal[1]>0):normal=-normal
 yy,xx=np.nonzero(mask[::2,::2]);coords=np.stack([xx*2,yy*2],1).astype(float)-np.array(low);t=(coords@axis)/length;r=(coords@normal)/length;bins=np.floor((t-.1)/(.8/32)).astype(int);valid=(bins>=0)&(bins<32)&(r>=-.15)&(r<=.5);bins=bins[valid];r=r[valid];profile=np.full(32,np.nan)
 for k in range(32):
  values=r[bins==k]
  if len(values)>=5:profile[k]=np.quantile(values,.98)
 good=np.flatnonzero(np.isfinite(profile))
 if len(good)<28 or good[0]>1 or good[-1]<30 or np.diff(good).max(initial=0)>3:raise ValueError('insufficient_outline_bins')
 profile=np.interp(np.arange(32),good,profile[good]);centers=np.linspace(.1+.8/64,.9-.8/64,32);first=float(profile[:3].mean());last=float(profile[-3:].mean());line=first+(last-first)*(centers-centers[:3].mean())/(centers[-3:].mean()-centers[:3].mean());residual=(profile-line).astype(np.float32)
 return residual,dict(raw_profile=profile.tolist(),affine_reference=line.tolist(),torso_length_px=length,valid_bins=len(good),normal=normal.tolist(),axis=axis.tolist(),low=list(low),high=list(high))
