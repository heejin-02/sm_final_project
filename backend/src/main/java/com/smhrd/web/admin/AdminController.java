package com.smhrd.web.admin;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import io.swagger.v3.oas.annotations.Operation;

@RestController
@RequestMapping("/api/admin/users")
public class AdminController {

    @Autowired
    private AdminService adminService;

    @Operation(summary = "전체 회원 목록 조회", description = "회원 및 농장 정보를 포함한 전체 회원 리스트를 조회합니다.")
    @GetMapping("/list")
    public ResponseEntity<List<AdminDTO>> getUserList() {
        System.out.println("[AdminUserController] /api/admin/users/list 호출됨");

        List<AdminDTO> userList = adminService.getAllUsers();
        System.out.println("[AdminUserController] 조회된 회원 수: " + userList.size());

        if (userList.isEmpty()) {
            return ResponseEntity.noContent().build();
        }

        return ResponseEntity.ok(userList);
    }
    
    
    @Operation(summary = "회원 상세 조회", description = "회원 전화번호를 통해 상세 정보를 조회합니다.")
    @GetMapping("/{userPhone}")
    public ResponseEntity<List<AdminDTO>> getUserDetail(@PathVariable String userPhone) {
        System.out.println("[AdminController] /api/admin/users/" + userPhone + " 호출됨");
        List<AdminDTO> userDetailList = adminService.selectUserByPhone(userPhone);

        if (userDetailList == null || userDetailList.isEmpty()) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(userDetailList);
    }

    // 농장 수정
    @Operation(summary = "농장 정보 수정", description = "farm_idx를 활용해 농장정보를 수정")
    @PutMapping("/farms/{farmIdx}")
    public ResponseEntity<String> updateFarm(@PathVariable Long farmIdx, @RequestBody AdminDTO farm) {
        farm.setFarmIdx(farmIdx);
        boolean success = adminService.updateFarmInfo(farm);
        if(success) return ResponseEntity.ok("농장 정보 수정 완료");
        else return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("수정 실패");
    }

    // 농장 삭제
    @Operation(summary = "농장 정보 삭제", description = "farm_idx를 활용해 농장정보를 삭제")
    @DeleteMapping("/farm/{farmIdx}")
    public ResponseEntity<String> deleteFarm(@PathVariable Long farmIdx) {
        try {
            adminService.deleteFarm(farmIdx);
            return ResponseEntity.ok("삭제 성공");
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("삭제 실패: " + e.getMessage());
        }
    }



    

}
