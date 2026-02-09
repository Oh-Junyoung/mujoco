import pickle
from pathlib import Path

class TopologyDataLoader:
    """pkl 파일에서 기구 위상 정보를 로드하는 클래스"""
    
    def __init__(self, data_folder='data', filename='topology_info.pkl'):
        """
        Args:
            data_folder: 데이터 폴더 이름 (기본값: 'data')
            filename: pkl 파일 이름 (기본값: 'topology_info.pkl')
        """
        self.data_folder = Path(data_folder)
        self.filename = filename
        self.filepath = self.data_folder / filename
        self.data = None
    
    def load(self):
        """pkl 파일을 로드"""
        try:
            with open(self.filepath, 'rb') as f:
                self.data = pickle.load(f)
            print(f"✓ 파일 로드 성공: {self.filepath}")
            return self.data
        except FileNotFoundError:
            print(f"✗ 파일을 찾을 수 없습니다: {self.filepath}")
            raise
        except Exception as e:
            print(f"✗ 파일 로드 중 오류 발생: {e}")
            raise
    
    def show_structure(self):
        """데이터 구조를 출력"""
        if self.data is None:
            print("먼저 load() 메서드를 실행하세요.")
            return
        
        print("\n=== 데이터 구조 ===")
        if isinstance(self.data, dict):
            for key, value in self.data.items():
                print(f"\n[{key}]")
                print(f"  타입: {type(value).__name__}")
                if isinstance(value, (list, tuple)):
                    print(f"  길이: {len(value)}")
                    print(f"  내용: {value}")
                elif hasattr(value, 'shape'):  # numpy array
                    print(f"  Shape: {value.shape}")
                    print(f"  내용:\n{value}")
                else:
                    print(f"  값: {value}")
        else:
            print(f"타입: {type(self.data).__name__}")
            print(f"내용: {self.data}")
    
    def get(self, key=None):
        """특정 키의 데이터를 가져오기"""
        if self.data is None:
            print("먼저 load() 메서드를 실행하세요.")
            return None
        
        if key is None:
            return self.data
        
        if isinstance(self.data, dict):
            return self.data.get(key)
        else:
            print("데이터가 dictionary 형태가 아닙니다.")
            return None