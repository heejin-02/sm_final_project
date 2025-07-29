package com.smhrd.web.user;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
 
import io.swagger.v3.oas.annotations.Operation;

@RestController
@RequestMapping("/api/user")
public class UserController {
	
	@Autowired
    private UserService userService;

	@Operation(summary = "유저 핸드폰번호로 농장 리스트 조회")
	@GetMapping("/farms")
	public ResponseEntity<List<UserDTO>> getUserFarms(@RequestParam String userPhone) {
	    System.out.println("조회 유저폰: " + userPhone);
	    List<UserDTO> farms = userService.getFarmsByUserPhone(userPhone);
	    return ResponseEntity.ok(farms);
	}



    
}
