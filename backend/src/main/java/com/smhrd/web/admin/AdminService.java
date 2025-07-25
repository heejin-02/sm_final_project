package com.smhrd.web.admin;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class AdminService {

    @Autowired
    private AdminMapper adminMapper;

    public List<AdminDTO> getAllUsers() {
        System.out.println("[AdminUserService] getAllUsers() 호출됨");
        List<AdminDTO> userList = adminMapper.selectAllUsers();
        System.out.println("[AdminUserService] 조회된 회원 수: " + userList.size());
        return userList;
    }

    public List<AdminDTO> selectUserByPhone(String userPhone) {
        System.out.println("[AdminService] getUserByPhone() 호출됨: " + userPhone);
        return adminMapper.selectUserByPhone(userPhone);
    }
    
    // 수정
    public boolean updateFarmInfo(AdminDTO farm) {
        try {
            adminMapper.updateFarmInfo(farm);
            return true;
        } catch(Exception e) {
            e.printStackTrace();
            return false;
        }
    }
    // 삭제
    public void deleteFarm(Long farmIdx) {
        // 자식 테이블 삭제 쿼리 없이 부모 테이블 삭제만 호출
        adminMapper.deleteFarm(farmIdx);
    }


}
