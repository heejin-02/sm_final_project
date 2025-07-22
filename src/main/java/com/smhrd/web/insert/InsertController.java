package com.smhrd.web.insert;

import org.springframework.web.bind.annotation.GetMapping;

public class InsertController {
    
	@GetMapping("/insert")
	public String insert() {
		return "insert";
	}

}
