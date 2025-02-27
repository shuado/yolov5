import requests
from threading import Thread

# 事件触发器
def trigger_events(image_path, similarity):
    """多线程执行事件触发"""
    # 示例1：调用Webhook
    # Thread(target=send_webhook, args=(image_path, similarity)).start()
    
    # 示例2：保存到数据库
    # Thread(target=log_to_database, args=(image_path, similarity)).start()

def send_webhook(img_path, sim):
    """发送告警到指定API"""
    files = {'image': open(img_path, 'rb')}
    data = {'similarity': sim, 'camera_id': 'CAM01'}
    requests.post('http://your-api/alert', files=files, data=data)

def log_to_database(img_path, sim):
    """SQLite日志记录"""
    import sqlite3
    conn = sqlite3.connect('alerts.db')
    c = conn.cursor()
    c.execute('''INSERT INTO alerts (time, similarity, image) 
                 VALUES (datetime('now'), ?, ?)''', (sim, img_path))
    conn.commit()
    conn.close()