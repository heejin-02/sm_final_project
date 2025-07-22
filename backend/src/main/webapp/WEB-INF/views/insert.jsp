<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>회원 추가 페이지</title>
</head>
<body>
	<h1>회원 등록</h1>
	<form action="insertForm" method="post">
		<ul>
			<li>
				<span>농장주 이름</span>
				<input type="text" name="farmOwner" placeholder="이름을 입력해주세요" class="ipt_tt" maxlength="100" required>
			
			</li>
			<li>
				<span>농장주 휴대폰번호</span>
				<input type="text" name="phone" placeholder="휴대폰 번호를 입력해주세요" class="ipt_tt" maxlength="100" required>
			</li>
			<li>
				<span>농장주 아이디</span>
				<input type="text" name="userId" placeholder="아이디를 입력해주세요" class="ipt_tt" maxlength="100" required>
			
			</li>
			<li>
				<span>농장주 비밀번호</span>
				<input type="text" name="password" placeholder="비밀번호를 입력해주세요" class="ipt_tt" maxlength="100" required>
			</li>
			<li>
				<span>농장주 비밀번호 확인</span>
				<input type="text" name="passwordCheck" placeholder="비밀번호를 다시 입력해주세요" class="ipt_tt" maxlength="100" required>
			
			</li>
			<li>
				<span>관리할 농장 이름</span>
				<input type="text" name="farmName" placeholder="농장이름을 입력해주세요" class="ipt_tt" maxlength="100" required>
			</li>
			<li>
				<span>관리할 농장 전화번호</span>
				<input type="text" name="farmPhone" placeholder="농장 전화번호를 입력해주세요" class="ipt_tt" maxlength="100" required>
			
			</li>
			<li>
				<span>관리할 농장 주소</span>
				<input type="text" name="farmAddress" placeholder="농장 주소를 입력해주세요" class="ipt_tt" maxlength="100" required>
			</li>
		</ul>
	</form> 
</body>
</html>