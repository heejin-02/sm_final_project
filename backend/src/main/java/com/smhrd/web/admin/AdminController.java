package com.smhrd.web.admin;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;

@Tag(name = "Admin Controller", description = "관리자 관련 API")
@RestController
@RequestMapping("/api/admin/users")
public class AdminController {

    @Autowired
    private AdminService adminService;
    
    @GetMapping("/list")
    @Operation(summary = "회원 및 농장 정보를 포함한 전체 회원 리스트를 조회합니다.")
    public ResponseEntity<List<AdminDTO>> getUserList() {
        System.out.println("[AdminUserController] /api/admin/users/list 호출됨");

        List<AdminDTO> userList = adminService.getAllUsers();
        System.out.println("[AdminUserController] 조회된 회원 수: " + userList.size());

        if (userList.isEmpty()) {
            return ResponseEntity.noContent().build();
        }

        return ResponseEntity.ok(userList);
    }
    
    @GetMapping("/{userPhone}")
    @Operation(summary = "회원 전화번호를 통해 상세 정보를 조회합니다.")
    public ResponseEntity<List<AdminDTO>> getUserDetail(@PathVariable String userPhone) {
        System.out.println("[AdminController] /api/admin/users/" + userPhone + " 호출됨");
        List<AdminDTO> userDetailList = adminService.selectUserByPhone(userPhone);

        if (userDetailList == null || userDetailList.isEmpty()) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(userDetailList);
    }
    
    @PutMapping("/farms/{farmIdx}")
    @Operation(summary = "farm_idx를 활용해 농장정보를 수정합니다.")
    public ResponseEntity<String> updateFarm(@PathVariable Long farmIdx, @RequestBody AdminDTO farm) {
        farm.setFarmIdx(farmIdx);
        boolean success = adminService.updateFarmInfo(farm);
        if(success) return ResponseEntity.ok("농장 정보 수정 완료");
        else return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("수정 실패");
    }

    @DeleteMapping("/farm/{farmIdx}")
    @Operation(summary = "farm_idx를 활용해 농장정보를 삭제합니다.")
    public ResponseEntity<String> deleteFarm(@PathVariable Long farmIdx) {
        try {
            adminService.deleteFarm(farmIdx);
            return ResponseEntity.ok("삭제 성공");
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("삭제 실패: " + e.getMessage());
        }
    }
    
    @PutMapping("/{userPhone}")
    @Operation(summary = "userPhone 기준으로 회원 이름, 비밀번호 수정합니다.")
    public ResponseEntity<String> updateUser(@PathVariable String userPhone, @RequestBody AdminDTO user) {
        user.setUserPhone(userPhone);
        boolean success = adminService.updateUserInfo(user);
        if(success) return ResponseEntity.ok("회원 정보 수정 완료");
        else return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("회원 정보 수정 실패");
    }
    
    @DeleteMapping("/{userPhone}")
    @Operation(summary = "userPhone 기준으로 회원 정보 삭제합니다.")
    public ResponseEntity<String> deleteUser(@PathVariable String userPhone) {
        boolean success = adminService.deleteUser(userPhone);
        if(success) return ResponseEntity.ok("회원 삭제 완료");
        else return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("회원 삭제 실패");
    }




    

}
